import logging
import time
import numpy as np  # Keep numpy for potential future use with scores
from typing import List, Tuple, Optional, Dict, Any  # Added Any
from .base import BaseGenerator
from ..utils import (
    client,
    get_prompt_logprobs,
    brushup_statement_ending,
    get_token_ids,
)  # Import the Together client directly and get_prompt_logprobs

logger = logging.getLogger(__name__)  # Get logger specific to this module


class BeamSearchGenerator(BaseGenerator):
    """
    Statement generator using Beam Search with Egalitarian Welfare.
    Maintains a fixed number (beam_width) of candidate sequences at each step.
    Expands candidates and evaluates them based on the minimum reward (log probability)
    assigned by agent-specific perspectives. Selects beams maximizing this minimum reward.
    """

    # --- Constants ---
    # Using Llama 3 specific tokens based on documentation/common usage
    LLAMA3_EOS_TOKENS = {
        "<|eot_id|>",
        "<|end_of_text|>",
        ".\n\n",
        ".\n",
        "\n\n",
        '."\n\n',
        "<end_of_turn>",
        "<eos>",
    }

    # Tokens to bias against during generation
    BIAS_AGAINST_TOKENS = [
        "...",
        '"',
        "***",
        "**",
        "\n\n\n",
        "\n\n\n\n",
        ":",
        " ...",
        " .",
        " •",
        "<end_of_turn>",
        "<eos>",
        "<start_of_turn>",
        # EOS tokens removed from bias list
    ]

    # Default bias value (large negative number to strongly discourage the tokens)
    DEFAULT_BIAS_VALUE = -1000000
    # Adapted from FiniteLookahead reference prompts
    DEFAULT_REFERENCE_SYSTEM_PROMPT = """You are generating a consensus statement that represents the views of multiple participants.
Your task is to continue the statement in a way that addresses the issue and considers all participants' opinions. Be concise and keep the statement short (less than 50 tokens) and focused. ONLY WRITE THE STATEMENT AND NOTHING ELSE."""
    DEFAULT_AGENT_SYSTEM_PROMPT_TEMPLATE = """You are generating a statement that represents the views of a single participant.
Your task is to continue the statement in a way that addresses the issue and considers ONLY this participant's opinion. Be concise and keep the statement short (less than 50 tokens) and focused. ONLY WRITE THE STATEMENT AND NOTHING ELSE."""

    DEFAULT_REFERENCE_USER_PROMPT_TEMPLATE = """Issue:
{issue}

Participants' opinions:
{opinions_text}

Consensus statement (less than 50 tokens):
"""
    # Added Agent User Prompt Template
    DEFAULT_AGENT_USER_PROMPT_TEMPLATE = """Issue:
{issue}

Participant's opinion:
{opinion}

Statement reflecting ONLY this participant's opinion (less than 50 tokens):
"""

    def __init__(self, model_identifier: str, config: dict):
        """Initializes the BeamSearchGenerator."""
        super().__init__(model_identifier, config)
        # Configure logger level for this specific generator instance
        log_level_str = config.get("log_level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logger.setLevel(log_level)
        logger.info(f"BeamSearchGenerator logger level set to: {log_level_str}")

        # Get configuration parameters specific to Beam Search
        self.beam_width = self.config.get("beam_width", 3)
        self.max_tokens = self.config.get("max_tokens", 50)
        self.api_delay = self.config.get("api_delay", 0.1)  # Delay between API calls
        self.seed = self.config.get("seed")  # Optional base seed
        # Add a config for max sampling attempts per beam per step
        self.max_sampling_attempts = self.config.get(
            "max_sampling_attempts", self.beam_width
        )  # Default to beam_width
        # Beta parameter (optional, could be used to scale rewards if needed later)
        self.beta = self.config.get("beta", 1.0)

        # Add configuration for brushup option
        self.brushup = self.config.get("brushup", False)
        self.brushup_model = self.config.get(
            "brushup_model", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        )

        # Token biasing configuration
        self.use_token_biasing = self.config.get("use_token_biasing", True)  # Enable by default
        self.bias_value = self.config.get("bias_value", self.DEFAULT_BIAS_VALUE)
        self.bias_against_tokens = self.config.get("bias_against_tokens", self.BIAS_AGAINST_TOKENS)

        # If additional tokens were provided in config, extend the default list
        if "additional_bias_tokens" in self.config:
            self.bias_against_tokens.extend(self.config["additional_bias_tokens"])

        logger.info(f"  Beam Width: {self.beam_width}")
        logger.info(f"  Max Tokens: {self.max_tokens}")
        logger.info(f"  API Delay: {self.api_delay} seconds")
        logger.info(f"  Seed: {self.seed}")
        logger.info(
            f"  Max Sampling Attempts per Step/Beam: {self.max_sampling_attempts}"
        )
        logger.info(f"  Beta (Reward Scaling): {self.beta}")  # Log beta
        if self.brushup:
            logger.info(f"  Statement brushup enabled with model: {self.brushup_model}")
        if self.use_token_biasing:
            logger.info(f"  Token biasing enabled with bias value: {self.bias_value}")
            logger.debug(f"  Biasing against tokens: {self.bias_against_tokens}")

    def _create_reference_prompt(
        self, issue: str, agent_opinions: dict, current_statement: str
    ) -> tuple[str, str]:
        """
        Creates the system and user prompts for the reference API call (sampling).
        Renamed from _create_prompt for clarity.
        """
        system_prompt = self.DEFAULT_REFERENCE_SYSTEM_PROMPT
        opinions_text = "\n\n".join(
            [
                f"Participant {i+1}: {opinion}"
                for i, opinion in enumerate(agent_opinions.values())
            ]
        )
        # Start with the base template
        user_prompt = self.DEFAULT_REFERENCE_USER_PROMPT_TEMPLATE.format(
            issue=issue, opinions_text=opinions_text
        )
        # Append the current statement to the user prompt if it exists
        if current_statement:
            # Use simple concatenation, assuming model handles continuation correctly
            user_prompt = f"{user_prompt}{current_statement}"

        return system_prompt, user_prompt

    # --- Added Method: Create Agent-Specific Prompts ---
    def _create_agent_prompts(
        self, issue: str, agent_opinions: dict, current_statement: str
    ) -> List[Tuple[str, str]]:
        """
        Creates system and user prompts for each agent's perspective.

        Args:
            issue: The central issue.
            agent_opinions: Dictionary of agent opinions.
            current_statement: The statement generated so far.

        Returns:
            A list of tuples, where each tuple contains (system_prompt, user_prompt)
            for a specific agent.
        """
        agent_prompts = []
        for (
            agent_id,
            opinion,
        ) in agent_opinions.items():  # Use items() to potentially use agent_id later
            system_prompt = (
                self.DEFAULT_AGENT_SYSTEM_PROMPT_TEMPLATE
            )  # Use agent template
            # Format user prompt for the specific agent
            user_prompt = self.DEFAULT_AGENT_USER_PROMPT_TEMPLATE.format(
                issue=issue, opinion=opinion
            )
            # Append the current statement
            if current_statement:
                user_prompt = f"{user_prompt}{current_statement}"
            agent_prompts.append((system_prompt, user_prompt))
        return agent_prompts

    # --- End Added Method ---

    def _append_to_statement(self, current_statement: str, next_token: str) -> str:
        """
        Appends the next token to the current statement.
        Simple concatenation for now.
        """
        return f"{current_statement}{next_token}"

    def _sample_next_tokens(
        self,
        system_prompt: str,  # Added system prompt argument
        user_prompt: str,  # Renamed from prompt
        num_desired_tokens: int,
        max_attempts: int,
        temperature: float,
        base_seed: Optional[int],
    ) -> List[Tuple[str, float]]:
        """
        Repeatedly samples single tokens using the REFERENCE prompt until
        num_desired_tokens unique tokens are found, or max_attempts is reached.

        Args:
            system_prompt: The system prompt string.
            user_prompt: The user prompt string.
            num_desired_tokens: The target number of unique tokens.
            max_attempts: Maximum number of API calls to make.
            temperature: Sampling temperature.
            base_seed: Optional base seed for reproducibility across attempts.

        Returns:
            A list of unique (token_string, reference_log_probability) tuples found.
            Note: The logprob returned here is from the reference model call.
        """
        collected_tokens: Dict[str, float] = {}  # {token_str: reference_logprob}
        attempts = 0

        logger.debug(
            f"      Attempting to sample {num_desired_tokens} unique tokens (max attempts: {max_attempts})..."
        )

        # Combine prompts for the API call
        full_prompt = (
            f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        )

        # Setup token biasing if enabled
        logit_bias = None
        if self.use_token_biasing and self.bias_against_tokens:
            logit_bias = {}
            for token in self.bias_against_tokens:
                token_ids = get_token_ids(self.model_identifier, token)
                if token_ids:
                    # Find token(s) containing the specified string
                    matching_tokens = {t: token_id for t, token_id in token_ids.items() if token in t}
                    if matching_tokens:
                        # Apply a bias to discourage the model from generating these tokens
                        for token_id in matching_tokens.values():
                            logit_bias[str(token_id)] = self.bias_value
                        logger.debug(
                            f"Applied bias {self.bias_value} to token '{token}': {matching_tokens}"
                        )

        while len(collected_tokens) < num_desired_tokens and attempts < max_attempts:
            attempts += 1
            attempt_seed = base_seed + attempts if base_seed is not None else None

            try:
                response = client.completions.create(
                    model=self.model_identifier,
                    prompt=full_prompt,
                    max_tokens=1,
                    temperature=temperature,
                    seed=attempt_seed,
                    logprobs=1,  # Request logprob for the generated token
                    logit_bias=logit_bias,  # Apply token biasing if enabled
                    stream=False,
                )

                if self.api_delay > 0:
                    time.sleep(self.api_delay)

                # --- Extract token and logprob (adapting for chat/completions) ---
                next_token = None
                token_logprob = None

                if hasattr(response, "choices") and response.choices:
                    choice = response.choices[0]
                    if (
                        hasattr(choice, "logprobs")
                        and choice.logprobs
                        and hasattr(choice.logprobs, "content")
                        and choice.logprobs.content
                    ):
                        # Chat Completions API structure
                        logprob_info = choice.logprobs.content[0]
                        next_token = logprob_info.token
                        token_logprob = logprob_info.logprob
                    elif (
                        hasattr(choice, "logprobs")
                        and choice.logprobs
                        and hasattr(choice.logprobs, "tokens")
                        and choice.logprobs.tokens
                    ):
                        # Completions API structure (legacy, check fields)
                        if choice.logprobs.tokens and choice.logprobs.token_logprobs:
                            next_token = choice.logprobs.tokens[0]
                            token_logprob = choice.logprobs.token_logprobs[0]

                # --- Process extracted token ---
                if next_token is not None and token_logprob is not None:
                    # Add to collection only if it's a new token
                    if next_token not in collected_tokens:
                        collected_tokens[next_token] = token_logprob
                        logger.debug(
                            f"        Attempt {attempts}: Found new token '{next_token}' (ref_logprob: {token_logprob:.4f}). Total unique: {len(collected_tokens)}"
                        )
                    else:
                        logger.debug(
                            f"        Attempt {attempts}: Found duplicate token '{next_token}'."
                        )
                else:
                    logger.warning(
                        f"        Attempt {attempts}: Could not extract token/logprob from response. Response structure might be unexpected."
                    )
                    # Optionally log the response structure for debugging
                    # logger.debug(f"Response object: {response}")

            except Exception as e:
                logger.error(
                    f"        Attempt {attempts}: Error during API call: {e}",
                    exc_info=False,  # Keep log less verbose unless DEBUG level
                )

        if len(collected_tokens) < num_desired_tokens:
            logger.warning(
                f"      Finished sampling after {attempts} attempts, but only found {len(collected_tokens)} unique tokens (target was {num_desired_tokens})."
            )
        else:
            logger.debug(
                f"      Successfully sampled {len(collected_tokens)} unique tokens in {attempts} attempts."
            )

        return list(collected_tokens.items())

    def _get_agent_token_logprob(
        self,
        agent_system_prompt: str,
        agent_user_prompt: str,  # This prompt should end *before* the token to evaluate
        token_to_evaluate: str,
        seed: Optional[int],
    ) -> Optional[float]:
        """
        Gets the log probability assigned by the agent's perspective to adding token_to_evaluate
        after the agent_user_prompt. Calculates this by comparing logprobs of the prompt
        with and without the token.

        Args:
            agent_system_prompt: The system prompt for the agent.
            agent_user_prompt: The user prompt representing the context *before* the token.
            token_to_evaluate: The specific token whose log probability contribution is needed.
            seed: Optional seed for the API call consistency.

        Returns:
            The summed log probability of the actual token(s) corresponding to
            token_to_evaluate when appended, or a default fallback value if it cannot be obtained.
        """
        try:
            full_user_prompt = self._append_to_statement(
                agent_user_prompt, token_to_evaluate
            )
            full_seed = seed + 1 if seed is not None else None  # Ensure different seed

            # Setup token biasing parameters to use with get_prompt_logprobs
            bias_against_tokens = self.bias_against_tokens if self.use_token_biasing else None
            bias_value = self.bias_value if self.use_token_biasing else None

            # Note: We're not using token biasing here since we're just getting logprobs for evaluation,
            # not generating new tokens. However, the parameters are passed for consistency in case
            # future implementations of get_prompt_logprobs integrate biasing.
            full_logprob_result = get_prompt_logprobs(
                model=self.model_identifier,
                system_prompt=agent_system_prompt,
                user_prompt=full_user_prompt,
                seed=full_seed,
            )
            if self.api_delay > 0:
                time.sleep(self.api_delay)

            if not full_logprob_result or not full_logprob_result[0]:
                logger.warning(
                    f"Could not get full logprobs for agent prompt ending with: ...{full_user_prompt[-50:]}"
                )
                # Instead of raising an error, return a fallback value
                return -10.0  # Reasonable fallback value for a low log probability

            full_tokens, full_logprobs = full_logprob_result

            # The logprobs for the added tokens are the last 'num_added_tokens' ones
            added_logprobs = full_logprobs[-1:]
            summed_logprob = sum(added_logprobs)

            logger.debug(
                f"      Agent Logprob for '{token_to_evaluate}': Sum({added_logprobs}) = {summed_logprob:.4f} (1 token)"
            )
            return summed_logprob

        except Exception as e:
            logger.error(
                f"Error getting agent logprob for token '{token_to_evaluate}': {e}",
                exc_info=False,
            )
            # Instead of re-raising the exception, return a fallback value
            logger.warning(f"Using fallback log probability value for token '{token_to_evaluate}'")
            return -10.0  # Reasonable fallback value for a low log probability

    # --- End Updated Method ---

    def generate_statement(self, issue: str, agent_opinions: dict) -> str:
        """
        Generates a statement using Beam Search logic with Egalitarian Welfare scoring.

        Args:
            issue: The central issue or topic.
            agent_opinions: A dictionary mapping agent identifiers to their opinions.

        Returns:
            The highest-scoring generated consensus statement based on min agent reward.
        """
        logger.info(
            f"Generating statement for '{issue}' using {self.__class__.__name__} (Egalitarian Welfare)"
        )
        logger.debug(f"  Full Config: {self.config}")

        if not client:  # Check if client is available (imported from utils)
            logger.error("Together client not initialized.")
            return "[ERROR: CLIENT NOT INITIALIZED]"

        num_agents = len(agent_opinions)
        if num_agents == 0:
            logger.warning("No agent opinions provided.")
            return ""

        # Beams store: (sequence_string, list_of_cumulative_agent_rewards)
        initial_rewards = [0.0] * num_agents
        beams = [("", initial_rewards)]
        # Completed sequences store: (sequence, final_agent_rewards_list)
        completed_sequences: List[Tuple[str, List[float]]] = []

        for step in range(self.max_tokens):
            logger.debug(f"\n--- Beam Search Step {step + 1}/{self.max_tokens} ---")
            # Candidates store: (new_sequence, new_cumulative_agent_rewards, last_token)
            candidates: List[Tuple[str, List[float], str]] = []
            # Base seed for all sampling attempts in this step
            step_base_seed = (
                self.seed
                + step
                * self.max_sampling_attempts
                * len(beams)
                * (num_agents + 1)  # Increase seed spacing
                if self.seed is not None
                else None
            )

            if not beams:
                logger.info("No active beams left to expand. Stopping search early.")
                break

            total_sampling_api_calls = 0
            total_scoring_api_calls = 0  # Will increase due to new logprob method

            # --- Iterate through current beams ---
            for beam_index, (current_seq, current_agent_rewards) in enumerate(beams):
                logger.debug(
                    f"  Expanding beam {beam_index+1}/{len(beams)}: '{current_seq}' (Min Reward: {min(current_agent_rewards):.4f})"
                )

                # --- 1. Sample Candidate Tokens using Reference Prompt ---
                ref_system_prompt, ref_user_prompt = self._create_reference_prompt(
                    issue, agent_opinions, current_seq
                )
                sampling_seed = (
                    step_base_seed + beam_index if step_base_seed is not None else None
                )

                sampled_tokens = self._sample_next_tokens(
                    system_prompt=ref_system_prompt,
                    user_prompt=ref_user_prompt,
                    num_desired_tokens=self.beam_width,
                    max_attempts=self.max_sampling_attempts,
                    temperature=1.0,  # Or get from config
                    base_seed=sampling_seed,
                )

                # --- 2. Evaluate Sampled Tokens with Agent Prompts ---
                if not sampled_tokens:
                    logger.debug(
                        f"    No unique tokens sampled for beam {beam_index+1}. Skipping."
                    )
                    continue

                agent_prompts = self._create_agent_prompts(
                    issue, agent_opinions, current_seq
                )

                for (
                    next_token,
                    _,
                ) in sampled_tokens:  # We only need the token string now
                    new_seq = self._append_to_statement(current_seq, next_token)
                    new_agent_rewards = []

                    # Get logprob for adding this next_token from each agent's perspective
                    for agent_idx, (agent_sys_prompt, agent_user_prompt) in enumerate(
                        agent_prompts
                    ):
                        agent_eval_seed = (
                            # Ensure seeds differ for context vs full calls within _get_agent_token_logprob
                            # and across agents/tokens
                            sampling_seed
                            + (agent_idx + 1)
                            * self.max_sampling_attempts
                            * 2  # Multiply by 2 for the two calls inside
                            if sampling_seed is not None
                            else None
                        )

                        # Get logprob contribution of ONLY the next_token given agent context
                        # This now uses the more robust 2-API-call method
                        agent_token_logprob_sum = self._get_agent_token_logprob(
                            agent_system_prompt=agent_sys_prompt,
                            agent_user_prompt=agent_user_prompt,  # Prompt ends *before* next_token
                            token_to_evaluate=next_token,
                            seed=agent_eval_seed,
                        )

                        if agent_token_logprob_sum is None:
                            logger.warning(
                                f"      Failed to get logprob sum for token '{next_token}' from agent {agent_idx}. Assigning default low score."
                            )
                            # Use fallback instead of raising an error
                            agent_token_logprob_sum = -10.0  # Consistent with fallback in _get_agent_token_logprob

                        # Calculate cumulative reward for this agent using the summed logprob
                        cumulative_reward = (
                            current_agent_rewards[agent_idx] + agent_token_logprob_sum
                        )

                        new_agent_rewards.append(cumulative_reward)

                    # Add candidate if all agent scores were obtained (or handled)
                    # if valid_agent_scores: # Uncomment if skipping candidates with failed scores
                    candidates.append((new_seq, new_agent_rewards, next_token))
                    min_rew = min(new_agent_rewards)
                    logger.debug(
                        f"      -> Candidate: '{next_token}' -> New Seq: '{new_seq}' (Agent Rewards: {[f'{r:.2f}' for r in new_agent_rewards]}, Min: {min_rew:.4f})"
                    )

            logger.debug(
                f"  Made approx {total_sampling_api_calls} sampling calls and {total_scoring_api_calls} scoring calls this step."
            )

            # --- Pruning candidates based on Egalitarian Welfare (Min Reward) ---
            if not candidates:
                logger.warning("No candidates generated in this step. Stopping search.")
                break

            # Sort by the minimum reward across agents for each candidate sequence
            sorted_candidates = sorted(
                candidates, key=lambda x: min(x[1]), reverse=True
            )

            new_beams = []
            seen_sequences_in_step = set()
            logger.debug(
                f"  Selecting top {self.beam_width} from {len(sorted_candidates)} candidates based on min agent reward:"
            )

            for i, (seq, agent_rewards, last_token) in enumerate(sorted_candidates):
                # Avoid adding duplicate sequences to the next beam set
                if seq in seen_sequences_in_step:
                    continue

                is_complete = last_token in self.LLAMA3_EOS_TOKENS

                # Check completion first
                if is_complete:
                    min_final_reward = min(agent_rewards)
                    logger.info(
                        f"    Found completed sequence: '{seq}' (Min Reward: {min_final_reward:.4f}, All Rewards: {[f'{r:.2f}' for r in agent_rewards]})"
                    )
                    completed_sequences.append((seq, agent_rewards))
                    # Don't add completed sequences to the next beams
                else:
                    # If not complete and we still need beams, add it
                    if len(new_beams) < self.beam_width:
                        min_reward = min(agent_rewards)
                        logger.debug(
                            f"    Adding to next beams: '{seq}' (Min Reward: {min_reward:.4f}, All Rewards: {[f'{r:.2f}' for r in agent_rewards]})"
                        )
                        new_beams.append(
                            (seq, agent_rewards)
                        )  # Store sequence and its agent rewards
                        seen_sequences_in_step.add(seq)

                # Optimization: If we have enough beams AND found completed ones,
                # we might still check remaining candidates ONLY for completion.
                # If we have enough beams and no more completed ones are likely needed, we could break early.
                # For simplicity, we iterate through all sorted candidates for now.
                # if len(new_beams) >= self.beam_width and not any(c[2].strip() in self.LLAMA3_EOS_TOKENS for c in sorted_candidates[i+1:]):
                #     break # Potential optimization if completion check is expensive

            beams = new_beams  # Update beams for the next step

            # Log current top beams for the next step
            if beams:
                logger.info(f"--- Step {step + 1} Top Beams (Sorted by Min Reward) ---")
                # Sort beams again just for logging display clarity
                sorted_log_beams = sorted(beams, key=lambda x: min(x[1]), reverse=True)
                for i, (seq, agent_rewards) in enumerate(sorted_log_beams):
                    min_r = min(agent_rewards)
                    logger.info(
                        f"  {i+1}. (Min Score: {min_r:.4f}) '\033[1;32m{seq}\033[0m' (All: {[f'{r:.2f}' for r in agent_rewards]})"
                    )
            else:
                logger.info(
                    f"--- Step {step + 1}: No active beams selected for the next step. ---"
                )

        # --- Final Selection ---
        logger.debug(
            f"Adding {len(beams)} remaining active beams to completed sequences."
        )
        # Add remaining beams (which didn't hit EOS) to completed list for final comparison
        completed_sequences.extend(beams)

        if not completed_sequences:
            logger.warning("No sequences generated or completed.")
            return ""

        # Filter out sequences with less than 5 words
        filtered_sequences = []
        for seq, rewards in completed_sequences:
            # Count words by splitting on whitespace
            word_count = len(seq.strip().split())
            if word_count >= 5:
                filtered_sequences.append((seq, rewards))
            else:
                logger.info(f"Filtering out sequence with only {word_count} words: '{seq}'")

        # If all sequences were filtered out, fall back to the original sequences
        if not filtered_sequences and completed_sequences:
            logger.warning("All sequences were less than 5 words. Using original sequences.")
            filtered_sequences = completed_sequences

        # Sort all completed sequences by their minimum agent reward (egalitarian welfare)
        sorted_completed = sorted(
            filtered_sequences, key=lambda x: min(x[1]), reverse=True
        )

        logger.info(
            "\n--- Beam Search Final Results (Ranked by Egalitarian Welfare) ---"
        )
        logger.info(
            f"  Generated {len(completed_sequences)} potential sequences, {len(filtered_sequences)} after filtering (showing top 5):"
        )
        for i, (seq, agent_rewards) in enumerate(sorted_completed[:5]):
            min_score = min(agent_rewards)
            logger.info(
                f"  {i+1}. Min Score: {min_score:.4f} | All: {[f'{r:.2f}' for r in agent_rewards]} | Sequence: '{seq}'"
            )

        if not sorted_completed:
            logger.error("Critical error: No sequences available after final sort.")
            return "[ERROR: NO FINAL SEQUENCE]"

        # Select the sequence with the highest minimum score
        best_sequence, best_rewards = sorted_completed[0]
        best_min_score = min(best_rewards)
        logger.info(
            f"\nSelected best sequence (Highest Min Score: {best_min_score:.4f}):\n{best_sequence}"
        )
        logger.info(
            f"  (Agent scores for best sequence: {[f'{r:.2f}' for r in best_rewards]})"
        )

        final_statement = best_sequence.strip()  # Strip whitespace
        pre_brushup_statement = final_statement  # Save the statement before brushup

        # Apply brushup if enabled
        if self.brushup:
            logger.info("Applying statement ending brushup...")
            brushed_statement = brushup_statement_ending(
                final_statement, self.brushup_model
            )
            if brushed_statement != final_statement:
                logger.info("Statement ending was fixed during brushup")
                logger.info(f"Brushed statement:\n{brushed_statement}")
            else:
                logger.info("Statement ending was already well-formed (not modified)")
            final_statement = brushed_statement

        # Add the pre-brushup statement as a property
        self.pre_brushup_statement = pre_brushup_statement

        return final_statement
