import time
import logging
import numpy as np
from typing import Optional

from .base import BaseGenerator
from ..utils import generate_text, get_prompt_logprobs

logger = logging.getLogger(__name__)  # Get logger specific to this module


class BestOfNGenerator(BaseGenerator):
    """
    Statement generator selecting the best from N samples based on egalitarian welfare.

    Generates N candidate statements using a reference policy, calculates agent-specific
    average log probabilities for each full statement, and selects the statement that
    maximizes the minimum average log probability across all agents (egalitarian welfare).
    """

    # --- Constants (similar to FiniteLookahead) ---
    DEFAULT_REWARD = -10.0
    REWARD_CLIP_MIN = -20.0
    REWARD_CLIP_MAX = 20.0
    UTILITY_EPSILON = 1e-9
    LLAMA3_EOS_TOKENS = {"<|eot_id|>", "<|end_of_text|>"}  # Needed for cleaning

    # Default system prompts (can be overridden via config if needed)
    DEFAULT_REFERENCE_SYSTEM_PROMPT = """You are generating a consensus statement that represents the views of multiple participants.
Your task is to continue the statement in a way that addresses the issue and considers all participants' opinions. Be concise and keep the statement short (less than 50 tokens) and focused. ONLY WRITE THE STATEMENT AND NOTHING ELSE."""
    DEFAULT_AGENT_SYSTEM_PROMPT_TEMPLATE = """You are generating a statement that represents the views of a single participant.
Your task is to continue the statement in a way that addresses the issue and considers ONLY this participant's opinion. Be concise and keep the statement short (less than 50 tokens) and focused. ONLY WRITE THE STATEMENT AND NOTHING ELSE."""

    DEFAULT_AGENT_USER_PROMPT_TEMPLATE = """Issue: {issue}\n\nAgent's opinion:\n{opinion}\n\nStatement reflecting this opinion (less than 50 tokens): """
    DEFAULT_REFERENCE_USER_PROMPT_TEMPLATE = """Issue: {issue}\n\nParticipants' opinions:\n{opinions_text}\n\nConsensus statement (less than 50 tokens): """

    def __init__(self, model_identifier: str, config: dict):
        super().__init__(model_identifier, config)
        # Set logger level based on config for THIS generator
        log_level_str = config.get(
            "log_level", "INFO"
        ).upper()  # Default to INFO if not in method's config
        log_level = getattr(
            logging, log_level_str, logging.INFO
        )  # Convert string to logging level
        logger.setLevel(log_level)  # Set the level for THIS logger instance
        logger.info(f"BestOfNGenerator logger level set to: {log_level_str}")

        # Add configuration for API delay
        self.api_delay = self.config.get("api_delay", 0.1)  # Default to 0.1 seconds
        logger.info(f"  API call delay set to: {self.api_delay} seconds")

    def generate_statement(self, issue: str, agent_opinions: dict) -> str:
        """Generates N statements and selects the best one based on egalitarian welfare."""
        logger.info(
            f"Generating statement for '{issue}' using {self.__class__.__name__}"
        )
        logger.info(f"  Config: {self.config}")

        # Extract configuration parameters
        # Check for num_best_of_n first (used in config files), then fallback to n
        n = self.config.get(
            "num_best_of_n", self.config.get("n", 3)
        )  # Number of candidates to generate
        max_tokens = self.config.get("max_tokens", 50)
        beta = self.config.get("beta", 1.0)
        seed = self.config.get("seed")  # seed is Optional[int]
        temperature = self.config.get(
            "temperature", 1.0
        )  # Fixed from using "beta" incorrectly

        logger.info(f"  Number of candidates (n): {n}")
        logger.info(f"  Max tokens per candidate: {max_tokens}")
        logger.info(f"  Beta for reward scaling: {beta}")
        logger.info(f"  Temperature for generation: {temperature}")
        logger.info(f"  Seed: {seed}")

        # --- Format System Prompts ---
        formatted_reference_system_prompt = self.DEFAULT_REFERENCE_SYSTEM_PROMPT
        formatted_agent_system_prompts = {
            agent_id: self.DEFAULT_AGENT_SYSTEM_PROMPT_TEMPLATE.format(
                issue=issue, opinion=opinion
            )
            for agent_id, opinion in agent_opinions.items()
        }
        # --- End Prompt Formatting ---

        # --- Create Base User Prompt for Reference Policy ---
        opinions_text = "\n\n".join(
            [
                f"Participant {i+1}: {opinion}"
                for i, opinion in enumerate(agent_opinions.values())
            ]
        )
        reference_base_user_prompt = self.DEFAULT_REFERENCE_USER_PROMPT_TEMPLATE.format(
            issue=issue, opinions_text=opinions_text
        )
        logger.debug(f"Reference base user prompt: {reference_base_user_prompt}")
        # --- End Base Prompt ---

        # --- Step 1: Generate N Candidate Statements ---
        candidate_statements = []
        logger.info(f"Generating {n} candidate statements...")
        for i in range(n):
            gen_seed = seed + i if seed is not None else None  # Ensure different seeds
            logger.debug(f"Generating candidate {i+1}/{n} with seed {gen_seed}")
            try:
                statement = generate_text(
                    model=self.model_identifier,
                    system_prompt=formatted_reference_system_prompt,
                    user_prompt=reference_base_user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=gen_seed,
                    # Stop sequences might be useful here depending on model behavior
                    # terminators=list(self.LLAMA3_EOS_TOKENS),
                    use_chat_completions=True,  # Assuming chat model
                )
                # --- Add delay after API call ---
                if self.api_delay > 0:
                    time.sleep(self.api_delay)
                # --- End delay ---

                cleaned_statement = self._clean_generated_text(statement)
                if cleaned_statement:
                    candidate_statements.append(cleaned_statement)
                    logger.debug(f"  Candidate {i+1}: '{cleaned_statement}'")
                else:
                    logger.warning(
                        f"  Candidate {i+1} resulted in empty statement after cleaning."
                    )
                    # Optionally generate another one? For now, just skip.

            except Exception as e:
                logger.error(f"Error generating candidate {i+1}: {e}", exc_info=True)
                # Add a placeholder or skip? Skipping for now.

        if not candidate_statements:
            logger.error("No valid candidate statements were generated.")
            return "[ERROR: Failed to generate any candidates]"

        logger.info(f"Generated {len(candidate_statements)} valid candidates.")

        # --- Step 2: Calculate Rewards for Each Candidate ---
        # This requires getting logprobs for the base prompt and the prompt+candidate
        # for both the reference policy and each agent policy.
        reward_results = self._calculate_candidate_rewards(
            issue=issue,
            agent_opinions=agent_opinions,
            candidate_statements=candidate_statements,
            beta=beta,
            seed=seed,
            formatted_reference_system_prompt=formatted_reference_system_prompt,
            formatted_agent_system_prompts=formatted_agent_system_prompts,
            reference_base_user_prompt=reference_base_user_prompt,
        )
        agent_rewards_per_candidate = reward_results[
            "agent_rewards"
        ]  # Dict[agent_id, List[float]]

        # Check if reward calculation failed critically
        if not agent_rewards_per_candidate:
            logger.error("Reward calculation failed. Cannot determine best statement.")
            # Return the first candidate as a fallback? Or error?
            return (
                candidate_statements[0]
                if candidate_statements
                else "[ERROR: Reward calculation failed]"
            )

        # --- Step 3: Calculate Egalitarian Welfare ---
        egalitarian_welfare_scores = self._calculate_egalitarian_welfare(
            agent_rewards_per_candidate, candidate_statements
        )

        # --- Log Candidates and Egalitarian Scores ---
        logger.info("Calculated Egalitarian Welfare Scores for Candidates:")
        if len(candidate_statements) == len(egalitarian_welfare_scores):
            for i, (stmt, score) in enumerate(
                zip(candidate_statements, egalitarian_welfare_scores)
            ):
                logger.info(
                    f"  Candidate {i+1}: '{stmt[:80]}...' - Egalitarian Welfare: {score:.4f}"
                )
        else:
            logger.warning(
                "Mismatch between number of candidates and Egalitarian scores, cannot log details."
            )
        # --- End Logging ---

        # --- Step 4: Select Best Statement ---
        if len(egalitarian_welfare_scores) == 0:
            logger.warning(
                "No Egalitarian welfare scores calculated. Returning first candidate as fallback."
            )
            best_statement = candidate_statements[0]
        else:
            best_candidate_idx = np.argmax(egalitarian_welfare_scores)
            best_statement = candidate_statements[best_candidate_idx]
            logger.info(
                f"Selected best candidate index: {best_candidate_idx} (Score: {egalitarian_welfare_scores[best_candidate_idx]:.4f})"
            )

        logger.info(
            f"Final selected statement (using egalitarian welfare): {best_statement}"
        )
        return best_statement

    def _clean_generated_text(self, text: str) -> str:
        """Clean up generated text (similar to FiniteLookahead)."""
        if not text:
            return ""

        cleaned_text = text.strip()  # Start by stripping whitespace

        # Remove common instruction-following prefixes
        prefixes = [
            "Consensus statement:",
            "Statement:",
            "Here is the consensus statement:",
            "Here is a statement reflecting this opinion:",
            "Okay, here is the statement:",
            # Add other potential unwanted prefixes
        ]
        for prefix in prefixes:
            if cleaned_text.lower().startswith(prefix.lower()):
                # Find the start index of the actual content after the prefix
                start_index = cleaned_text.lower().find(prefix.lower()) + len(prefix)
                # Remove the prefix and leading/trailing whitespace that might remain
                cleaned_text = cleaned_text[start_index:].strip()
                break  # Assume only one prefix needs removal

        # Remove potential EOS tokens if they appear at the end
        for eos in self.LLAMA3_EOS_TOKENS:
            if cleaned_text.endswith(eos):
                cleaned_text = cleaned_text[: -len(eos)].strip()

        return cleaned_text

    def _calculate_candidate_rewards(
        self,
        issue: str,
        agent_opinions: dict,
        candidate_statements: list[str],
        beta: float,
        seed: Optional[int],
        formatted_reference_system_prompt: str,
        formatted_agent_system_prompts: dict[str, str],
        reference_base_user_prompt: str,  # Pass the base prompt
    ) -> dict:
        """
        Calculate average log probabilities for each agent for each candidate statement.

        Instead of calculating reward differences from a reference policy, we now directly
        compute the average log probability that each agent assigns to each candidate.
        Higher average log probabilities indicate better alignment with the agent's preferences.
        """
        logger.info(
            "Calculating agent average log probabilities for each candidate statement"
        )

        # Dictionary to store agent average logprobs for each candidate
        agent_avg_logprobs = {agent_id: [] for agent_id in agent_opinions}

        # --- Calculate Logprobs for Each Candidate for each Agent ---
        for i, candidate in enumerate(candidate_statements):
            path_seed_offset = i * len(agent_opinions)  # Unique offset per candidate

            # --- Agent Policy Logprobs for Candidates ---
            for agent_id, opinion in agent_opinions.items():
                agent_seed = (
                    seed + 300000 + path_seed_offset + hash(agent_id) % 100
                    if seed is not None
                    else None
                )
                agent_system_prompt = formatted_agent_system_prompts[agent_id]
                agent_base_user_prompt = self.DEFAULT_AGENT_USER_PROMPT_TEMPLATE.format(
                    issue=issue, opinion=opinion
                )
                # agent_full_user_prompt = f"{agent_base_user_prompt} {candidate.strip()}"

                system_prompt = f"{agent_system_prompt}\n\n{agent_base_user_prompt}"
                user_prompt = candidate

                try:
                    # Get log probabilities for the full prompt (base + candidate)
                    _, agent_full_logprobs = get_prompt_logprobs(
                        model=self.model_identifier,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=1.0,
                        seed=agent_seed,
                    )
                    # --- Add delay after API call ---
                    if self.api_delay > 0:
                        time.sleep(self.api_delay)
                    # --- End delay ---

                    # Extract candidate logprobs (tokens after the context)
                    candidate_logprobs = agent_full_logprobs

                    # Calculate average log probability if we have valid logprobs
                    valid_logprobs = [lp for lp in candidate_logprobs if lp is not None]
                    if valid_logprobs:
                        avg_logprob = sum(valid_logprobs) / len(valid_logprobs)
                        logger.debug(
                            f"Agent {agent_id}, Candidate {i+1}: Avg LogProb = {avg_logprob:.4f} (based on {len(valid_logprobs)} tokens)"
                        )
                        agent_avg_logprobs[agent_id].append(avg_logprob)
                    else:
                        logger.warning(
                            f"Agent {agent_id}, Candidate {i+1}: No valid logprobs found. Using default reward."
                        )
                        agent_avg_logprobs[agent_id].append(self.DEFAULT_REWARD)

                except Exception as e:
                    logger.error(
                        f"Failed to get logprobs for agent {agent_id}, candidate {i+1}: {e}",
                        exc_info=True,
                    )
                    agent_avg_logprobs[agent_id].append(self.DEFAULT_REWARD)

        # Log the average logprobs for debugging
        for agent_id, avg_logprobs in agent_avg_logprobs.items():
            logger.info(f"  Agent {agent_id} average logprobs: {avg_logprobs}")

        return {"agent_rewards": agent_avg_logprobs}

    def _calculate_egalitarian_welfare(
        self, agent_rewards: dict[str, list[float]], candidate_statements: list[str]
    ) -> np.ndarray:
        """
        Calculate the egalitarian welfare (minimum average log probability) for each candidate statement.

        For each candidate, we find the minimum average log probability across all agents.
        The goal is to select the candidate that maximizes this minimum value.
        """
        logger.info(
            "Calculating egalitarian welfare (min avg logprob) for each candidate statement"
        )

        if not agent_rewards:
            logger.warning(
                "Agent rewards dictionary is empty. Cannot calculate egalitarian welfare."
            )
            return np.array([])

        # Find number of candidates from the first agent's reward list
        try:
            num_candidates = len(next(iter(agent_rewards.values())))
        except StopIteration:
            logger.warning(
                "Agent rewards dictionary is empty. Cannot calculate egalitarian welfare."
            )
            return np.array([])

        if num_candidates == 0:
            logger.warning(
                "No candidates found in agent rewards. Cannot calculate egalitarian welfare."
            )
            return np.array([])
        if num_candidates != len(candidate_statements):
            logger.warning(
                f"Mismatch between number of rewards ({num_candidates}) and candidate statements ({len(candidate_statements)})."
            )
            # Proceeding, but logging might be off

        # Initialize storage for logprobs per candidate
        logprobs_per_candidate = {}
        logger.debug(f"Calculating egalitarian welfare for {num_candidates} candidates")

        for agent_id, avg_logprobs in agent_rewards.items():
            if len(avg_logprobs) != num_candidates:
                logger.error(
                    f"Agent {agent_id} has {len(avg_logprobs)} logprobs, expected {num_candidates}. Skipping agent in egalitarian calculation."
                )
                continue

            logprobs_array = np.array(avg_logprobs, dtype=float)
            if np.any(~np.isfinite(logprobs_array)):
                logger.warning(
                    f"Non-finite logprobs found for agent {agent_id}. Using defaults."
                )
                logprobs_array = np.nan_to_num(
                    logprobs_array,
                    nan=self.DEFAULT_REWARD,
                    posinf=self.REWARD_CLIP_MAX,
                    neginf=self.REWARD_CLIP_MIN,
                )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"  Agent {agent_id} avg logprobs: {logprobs_array}")

            # Store logprobs for each agent
            for i, logprob in enumerate(logprobs_array):
                if i not in logprobs_per_candidate:
                    logprobs_per_candidate[i] = []
                logprobs_per_candidate[i].append(logprob)

        # Calculate egalitarian welfare (minimum avg logprob) for each candidate
        egalitarian_welfare = np.zeros(num_candidates, dtype=float)
        for i in range(num_candidates):
            if i in logprobs_per_candidate and logprobs_per_candidate[i]:
                # The egalitarian welfare is the minimum avg logprob across all agents
                egalitarian_welfare[i] = min(logprobs_per_candidate[i])
            else:
                # If no logprobs for this candidate, assign a very low value
                egalitarian_welfare[i] = self.DEFAULT_REWARD

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"  Candidate {i} egalitarian welfare (min avg logprob): {egalitarian_welfare[i]}"
                )

        logger.info(
            f"Egalitarian welfare scores calculated (showing first 5): {egalitarian_welfare[:5]}"
        )
        return egalitarian_welfare
