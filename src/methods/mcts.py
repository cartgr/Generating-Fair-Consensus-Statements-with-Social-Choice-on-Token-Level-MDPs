import logging
import time
import numpy as np
import math
import random
from typing import List, Tuple, Optional, Dict, Any
from .base import BaseGenerator
from ..utils import (
    client,
    get_prompt_logprobs,
    brushup_statement_ending,
)

logger = logging.getLogger(__name__)  # Get logger specific to this module


# --- MCTS Node ---
class Node:
    """Node in the Monte Carlo Search Tree."""

    def __init__(
        self,
        statement: str,
        parent: Optional["Node"] = None,
        token: Optional[str] = None,
        is_terminal: bool = False,
    ):
        self.statement: str = statement
        self.parent: Optional["Node"] = parent
        self.token: Optional[str] = token  # Token that led to this node
        self.children: Dict[str, "Node"] = {}  # Map token_str -> child Node
        self.visits: int = 0
        # Sum of minimum agent rewards from simulations passing through this node
        self.total_reward: float = 0.0
        # Average minimum agent reward (total_reward / visits)
        self.value: float = 0.0
        # Store the immediate reward of the token (min agent logprob) when evaluated
        self.immediate_reward: Optional[float] = None
        # Potential next tokens (token_str, ref_logprob) from sampling
        self.untried_tokens: Optional[List[Tuple[str, float]]] = None
        self.is_terminal: bool = is_terminal  # If the token leading here was EOS


class MCTSGenerator(BaseGenerator):
    """
    Statement generator using Monte Carlo Tree Search with Egalitarian Welfare.
    Builds a search tree, balancing exploration and exploitation (using UCB1)
    to find sequences maximizing the minimum agent reward (log probability).
    """

    # --- Constants (Copied from BeamSearchGenerator) ---
    LLAMA3_EOS_TOKENS = {
        "<|eot_id|>",
        "<|end_of_text|>",
        ".\n\n",
        ".\n",
        "\n\n",
        '."\n\n',
    }
    DEFAULT_REFERENCE_SYSTEM_PROMPT = """You are generating a consensus statement that represents the views of multiple participants.
Your task is to continue the statement in a way that addresses the issue and considers all participants' opinions. Be concise and coherent. ONLY WRITE THE CONSENSUS STATEMENT AND NOTHING ELSE."""
    DEFAULT_AGENT_SYSTEM_PROMPT_TEMPLATE = """You are generating a statement that represents the views of a single participant.
Your task is to continue the statement in a way that addresses the issue and considers ONLY this participant's opinion. Be concise and coherent. ONLY WRITE THE CONSENSUS STATEMENT AND NOTHING ELSE."""
    DEFAULT_REFERENCE_USER_PROMPT_TEMPLATE = """Issue:
{issue}

Participants' opinions:
{opinions_text}

Consensus statement:
"""
    DEFAULT_AGENT_USER_PROMPT_TEMPLATE = """Issue:
{issue}

Participant's opinion:
{opinion}

Statement reflecting ONLY this participant's opinion:
"""

    def __init__(self, model_identifier: str, config: dict):
        """Initializes the MCTSGenerator."""
        super().__init__(model_identifier, config)
        log_level_str = config.get("log_level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logger.setLevel(log_level)
        logger.info(f"MCTSGenerator logger level set to: {log_level_str}")

        # Get configuration parameters specific to MCTS
        self.num_simulations = self.config.get("num_simulations", 50)
        self.exploration_constant = self.config.get(
            "exploration_constant", 1.414
        )  # sqrt(2)
        self.max_tokens = self.config.get("max_tokens", 100)
        self.api_delay = self.config.get("api_delay", 0.1)
        self.seed = self.config.get("seed")
        self.expansion_sample_width = self.config.get("expansion_sample_width", 5)
        self.max_sampling_attempts = self.config.get(
            "max_sampling_attempts", self.expansion_sample_width * 3
        )
        self.rollout_depth = self.config.get("rollout_depth", 10)
        # Add gamma parameter to control weighting between immediate and rollout rewards
        self.gamma = self.config.get("gamma", 0.99)  # Default to heavily favor rollout

        # Add configuration for brushup option
        self.brushup = self.config.get("brushup", False)
        self.brushup_model = self.config.get(
            "brushup_model", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        )

        logger.info(f"  Num Simulations per Step: {self.num_simulations}")
        logger.info(f"  Exploration Constant (C): {self.exploration_constant}")
        logger.info(f"  Max Tokens: {self.max_tokens}")
        logger.info(f"  API Delay: {self.api_delay} seconds")
        logger.info(f"  Seed: {self.seed}")
        logger.info(f"  Expansion Sample Width: {self.expansion_sample_width}")
        logger.info(
            f"  Max Sampling Attempts per Expansion: {self.max_sampling_attempts}"
        )
        logger.info(f"  Rollout Depth: {self.rollout_depth}")
        logger.info(f"  Gamma (Rollout Weight): {self.gamma}")
        if self.brushup:
            logger.info(f"  Statement brushup enabled with model: {self.brushup_model}")

    # --- Prompt Creation Methods (Copied from BeamSearchGenerator) ---
    def _create_reference_prompt(
        self, issue: str, agent_opinions: dict, current_statement: str
    ) -> tuple[str, str]:
        """Creates the system and user prompts for the reference API call (sampling)."""
        system_prompt = self.DEFAULT_REFERENCE_SYSTEM_PROMPT
        opinions_text = "\n\n".join(
            [
                f"Participant {i+1}: {opinion}"
                for i, opinion in enumerate(agent_opinions.values())
            ]
        )
        user_prompt = self.DEFAULT_REFERENCE_USER_PROMPT_TEMPLATE.format(
            issue=issue, opinions_text=opinions_text
        )
        if current_statement:
            user_prompt = f"{user_prompt}{current_statement}"
        return system_prompt, user_prompt

    def _create_agent_prompts(
        self, issue: str, agent_opinions: dict, current_statement: str
    ) -> List[Tuple[str, str]]:
        """Creates system and user prompts for each agent's perspective."""
        agent_prompts = []
        for agent_id, opinion in agent_opinions.items():
            system_prompt = self.DEFAULT_AGENT_SYSTEM_PROMPT_TEMPLATE
            user_prompt = self.DEFAULT_AGENT_USER_PROMPT_TEMPLATE.format(
                issue=issue, opinion=opinion
            )
            if current_statement:
                user_prompt = f"{user_prompt}{current_statement}"
            agent_prompts.append((system_prompt, user_prompt))
        return agent_prompts

    # --- Token Appending (Copied from BeamSearchGenerator) ---
    def _append_to_statement(self, current_statement: str, next_token: str) -> str:
        """Appends the next token to the current statement."""
        return f"{current_statement}{next_token}"

    # --- Token Sampling (Copied & slightly adapted from BeamSearchGenerator) ---
    def _sample_next_tokens(
        self,
        system_prompt: str,
        user_prompt: str,
        num_desired_tokens: int,
        max_attempts: int,
        temperature: float,
        base_seed: Optional[int],
    ) -> List[Tuple[str, float]]:
        """
        Repeatedly samples single tokens using the REFERENCE prompt until
        num_desired_tokens unique tokens are found, or max_attempts is reached.
        Returns list of (token_string, reference_log_probability).
        """
        collected_tokens: Dict[str, float] = {}
        attempts = 0
        logger.debug(
            f"      Attempting to sample {num_desired_tokens} unique tokens (max attempts: {max_attempts})..."
        )
        full_prompt = (
            f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
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
                    logprobs=1,
                    stream=False,
                )
                if self.api_delay > 0:
                    time.sleep(self.api_delay)

                next_token, token_logprob = None, None
                if hasattr(response, "choices") and response.choices:
                    choice = response.choices[0]
                    # Simplified extraction logic (assuming one structure works)
                    if (
                        hasattr(choice, "logprobs")
                        and choice.logprobs
                        and hasattr(choice.logprobs, "content")
                        and choice.logprobs.content
                    ):
                        logprob_info = choice.logprobs.content[0]
                        next_token = logprob_info.token
                        token_logprob = logprob_info.logprob
                    elif (
                        hasattr(choice, "logprobs")
                        and choice.logprobs
                        and hasattr(choice.logprobs, "tokens")
                        and choice.logprobs.tokens
                        and choice.logprobs.token_logprobs
                    ):
                        next_token = choice.logprobs.tokens[0]
                        token_logprob = choice.logprobs.token_logprobs[0]

                if next_token is not None and token_logprob is not None:
                    if next_token not in collected_tokens:
                        collected_tokens[next_token] = token_logprob
                        logger.debug(
                            f"        Attempt {attempts}: Found new token '{next_token}' (ref_logprob: {token_logprob:.4f}). Total unique: {len(collected_tokens)}"
                        )
                else:
                    logger.warning(
                        f"        Attempt {attempts}: Could not extract token/logprob from response."
                    )
            except Exception as e:
                logger.error(
                    f"        Attempt {attempts}: Error during API call: {e}",
                    exc_info=False,
                )

        if len(collected_tokens) < num_desired_tokens:
            logger.warning(
                f"      Finished sampling after {attempts} attempts, only found {len(collected_tokens)} unique tokens (target: {num_desired_tokens})."
            )
        return list(collected_tokens.items())

    def _get_agent_token_logprob(
        self,
        agent_system_prompt: str,
        agent_user_prompt_before_token: str,
        token_to_evaluate: str,
        seed: Optional[int],
    ) -> Optional[float]:
        """
        Gets the log probability assigned by the agent's perspective to adding token_to_evaluate
        after the agent_user_prompt_before_token.
        Returns the logprob of *only* the token_to_evaluate.
        """

        system_prompt = f"{agent_system_prompt}\n\n{agent_user_prompt_before_token}"
        user_prompt = token_to_evaluate

        try:
            # Get logprobs for the prompt *including* the token
            full_seed = seed + 1 if seed is not None else None
            full_logprob_result = get_prompt_logprobs(
                model=self.model_identifier,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                seed=full_seed,
            )
            if self.api_delay > 0:
                time.sleep(self.api_delay)

            if not full_logprob_result or not full_logprob_result[0]:
                logger.warning(
                    f"Could not get full logprobs for agent prompt ending with: ...{user_prompt[-50:]}"
                )
                return None

            full_tokens, full_logprobs = full_logprob_result

            # # --- Refined logic to handle multi-token 'token_to_evaluate' ---
            # # Get logprobs for the prompt *without* the token
            # base_seed = seed + 2 if seed is not None else None
            # base_logprob_result = get_prompt_logprobs(
            #     model=self.model_identifier,
            #     system_prompt=agent_system_prompt,
            #     user_prompt=agent_user_prompt_before_token,
            #     seed=base_seed,
            # )
            if self.api_delay > 0:
                time.sleep(self.api_delay)

            # if not base_logprob_result or not base_logprob_result[0]:
            #     logger.warning(
            #         f"Could not get base logprobs for agent prompt ending with: ...{agent_user_prompt_before_token[-50:]}"
            #     )
            #     return None

            # base_tokens, base_logprobs = base_logprob_result

            # Calculate the sum of logprobs for the base and full prompts
            sum_full_logprobs = sum(full_logprobs) if full_logprobs else -float("inf")
            # sum_base_logprobs = sum(base_logprobs) if base_logprobs else -float("inf")

            # The logprob of the added token(s) is the difference
            # Handle potential floating point inaccuracies or API issues
            if sum_full_logprobs == -float("inf"):
                logger.warning(
                    f"Could not reliably calculate logprob difference for token '{token_to_evaluate}'"
                )
                return None

            token_sequence_logprob = sum_full_logprobs

            logger.debug(
                f"      Agent Logprob for token(s) '{token_to_evaluate}': {token_sequence_logprob:.4f}"
            )
            return token_sequence_logprob

        except Exception as e:
            logger.error(
                f"Error getting agent logprob for token '{token_to_evaluate}': {e}",
                exc_info=False,
            )
            return None

    # --- New: Agent Sequence Logprob Calculation ---
    def _get_agent_sequence_logprob(
        self,
        agent_system_prompt: str,
        agent_user_prompt_full: str,
        seed: Optional[int],
    ) -> Optional[float]:
        """
        Gets the total log probability assigned by the agent's perspective to the
        entire agent_user_prompt_full sequence.
        """
        try:

            logprob_result = get_prompt_logprobs(
                model=self.model_identifier,
                system_prompt=agent_system_prompt,
                user_prompt=agent_user_prompt_full,
                seed=seed,
            )
            if self.api_delay > 0:
                time.sleep(self.api_delay)

            if not logprob_result or not logprob_result[0]:
                logger.warning(
                    f"Could not get logprobs for agent sequence ending with: ...{agent_user_prompt_full[-50:]}"
                )
                return None

            tokens, logprobs = logprob_result
            total_logprob = sum(logprobs) if logprobs else -float("inf")

            if total_logprob == -float("inf"):
                logger.warning(f"Logprobs list was empty for agent sequence.")
                return None

            logger.debug(f"      Agent Total Logprob for sequence: {total_logprob:.4f}")
            return total_logprob

        except Exception as e:
            logger.error(
                f"Error getting agent sequence logprob: {e}",
                exc_info=False,
            )
            return None

    # --- MCTS Core Methods ---

    def _ucb1(self, node: Node, parent_visits: int) -> float:
        """Calculates the UCB1 score for a node."""
        if node.visits == 0:
            # Favor unvisited nodes strongly
            logger.debug(
                f"        UCB1 for '{node.token}' (visits=0): inf"
            )  # Optional: Log infinite score
            return float("inf")
        if parent_visits == 0:  # Should not happen if root is visited first
            logger.warning(
                f"        Parent visits are 0 for node '{node.token}', using 1."
            )
            parent_visits = 1

        exploitation_term = node.value
        exploration_term = self.exploration_constant * math.sqrt(
            math.log(parent_visits) / node.visits
        )
        score = exploitation_term + exploration_term
        # logger.debug(f"        UCB1 for '{node.token}': {exploitation_term:.4f} + {exploration_term:.4f} = {score:.4f}") # Optional: Log score components
        return score

    def _select(self, node: Node) -> Node:
        """Selects a node to expand using UCB1."""
        logger.debug(
            f"    Starting Selection from node: '{node.statement}' (visits={node.visits})"
        )
        original_node = node
        while not node.is_terminal:
            # If node has untried tokens or hasn't been sampled yet
            if node.untried_tokens is None or len(node.untried_tokens) > 0:
                logger.debug(
                    f"    Node '{node.statement}' has untried actions or needs sampling. Selecting for expansion."
                )
                return node
            # If node is fully expanded but has no children (e.g., sampling failed)
            if not node.children:
                logger.warning(
                    f"    Node '{node.statement}' is fully expanded but has no children. Treating as terminal for selection."
                )
                node.is_terminal = (
                    True  # Mark as effectively terminal if expansion failed
                )
                return node  # Return the node itself as it cannot proceed further

            # --- Select best child using UCB1 ---
            best_child = None
            best_score = -float("inf")
            parent_visits = (
                node.visits
            )  # Visits of the current node (parent for children)
            logger.debug(
                f"      Calculating UCB1 for children of '{node.statement}' (parent visits={parent_visits}):"
            )

            # Ensure deterministic selection order if scores are equal (optional)
            child_items = sorted(node.children.items())

            for token, child in child_items:
                score = self._ucb1(child, parent_visits)
                logger.debug(
                    f"        Child '{token}' (v={child.visits}, val={child.value:.4f}) -> UCB1 Score: {score:.4f}"
                )
                if score > best_score:
                    best_score = score
                    best_child = child
                # Handle potential infinite scores - prefer first encountered infinite score
                elif score == float("inf") and best_score != float("inf"):
                    best_score = score
                    best_child = child

            if best_child is None:
                # This case might occur if all children have -inf score (e.g. failed evals)
                logger.error(  # Changed to error as this indicates a problem
                    f"    Could not select a best child for node '{node.statement}'. All child scores might be invalid. Treating as terminal."
                )
                node.is_terminal = True
                return node

            logger.debug(
                f"      Selected best child: '{best_child.token}' (Score: {best_score:.4f})"
            )
            node = best_child  # Move down the tree

        logger.debug(
            f"    Selection finished. Reached node: '{node.statement}' (Terminal: {node.is_terminal})"
        )
        return (
            node  # Return the terminal node reached or the node selected for expansion
        )

    # --- New: Rollout Method ---
    def _rollout(
        self,
        start_node: Node,
        issue: str,
        agent_opinions: dict,
        rollout_depth: int,
        seed: Optional[int],
    ) -> float:
        """
        Performs a rollout from start_node using the reference policy.
        Generates up to rollout_depth tokens.
        Returns the minimum total log probability assigned by agents to the final sequence.
        """
        logger.debug(
            f"      Starting rollout from: '{start_node.statement}' (depth={rollout_depth})"
        )
        current_statement = start_node.statement
        rollout_sequence = ""
        base_seed = seed if seed is not None else random.randint(0, 100000)

        ref_sys_prompt, ref_user_prompt = self._create_reference_prompt(
            issue, agent_opinions, current_statement
        )

        try:
            response = client.completions.create(
                model=self.model_identifier,
                prompt=f"{ref_sys_prompt}\n\n{ref_user_prompt}",
                max_tokens=rollout_depth,
                temperature=1.0,
                seed=base_seed,
                logprobs=None,
                stream=False,
            )

            if self.api_delay > 0:
                time.sleep(self.api_delay)

            if not response or not response.choices:
                logger.warning(f"Rollout failed to generate a response.")
                return -100.0

            rollout_sequence = response.choices[0].text
            # current_statement += rollout_sequence  # Append the rollout sequence to the current statement
            logger.debug(f"Rollout sequence: {rollout_sequence}")
            logger.debug(f"Current statement: {current_statement}")

        except Exception as e:
            logger.error(f"Rollout failed: {e}")
            return -100.0

        # for i in range(rollout_depth):
        #     rollout_step_seed = base_seed + i
        # ref_sys_prompt, ref_user_prompt = self._create_reference_prompt(
        #     issue, agent_opinions, current_statement
        # )
        # # Sample exactly one token using the reference prompt
        # # Use _sample_next_tokens with num_desired=1, max_attempts=1 for simplicity
        # # Or call client directly for potentially better control/efficiency
        # try:
        #     response = client.completions.create(
        #         model=self.model_identifier,
        #         prompt=(
        #             f"{ref_sys_prompt}\n\n{ref_user_prompt}"
        #             if ref_sys_prompt
        #             else ref_user_prompt
        #         ),
        #         max_tokens=rollout_depth,
        #         temperature=1.0,  # Use a moderate temperature for rollout
        #         seed=rollout_step_seed,
        #         logprobs=None,  # Don't need logprobs here
        #         stream=False,
        #     )
        #     if self.api_delay > 0:
        #         time.sleep(self.api_delay)

        #     next_token = None
        #     if hasattr(response, "choices") and response.choices:
        #         next_token = response.choices[0].text
        #     else:  # Handle cases like Anthropic models if needed
        #         if hasattr(response, "completion"):
        #             next_token = response.completion

        #     if next_token is None or next_token == "":
        #         logger.warning(
        #             f"        Rollout step {i+1}/{rollout_depth}: No token generated. Stopping rollout."
        #         )
        #         break

        #     # Clean up potential whitespace issues if necessary
        #     # next_token = next_token.strip() # Optional: depends on model behavior

        #     rollout_sequence += next_token
        #     current_statement += next_token
        #     logger.debug(
        #         f"        Rollout step {i+1}/{rollout_depth}: Sampled '{next_token}' -> Current: '{current_statement}'"
        #     )

        #     # Check for EOS
        #     if next_token.strip() in self.LLAMA3_EOS_TOKENS:
        #         logger.debug(
        #             f"        Rollout step {i+1}/{rollout_depth}: EOS token found. Stopping rollout."
        #         )
        #         break

        # except Exception as e:
        #     logger.error(
        #         f"        Rollout step {i+1}/{rollout_depth}: Error during API call: {e}",
        #         exc_info=False,
        #     )
        #     # Treat API error as failure for this rollout
        #     logger.warning(
        #         "        Assigning very low reward due to rollout API error."
        #     )
        #     return -100.0  # Return very low reward on error

        if not rollout_sequence:
            logger.debug("      Rollout generated no tokens (or failed immediately).")
            # If the start_node itself was terminal, this shouldn't be called.
            # If rollout failed immediately, assign low value.
            return -100.0  # Assign low value if no progress was made

        # --- Evaluate the final sequence from the rollout ---
        # final_statement = (
        #     current_statement  # The statement including the rollout tokens
        # )
        logger.debug(
            f"      Rollout finished. Evaluating final statement: '{rollout_sequence}'"
        )
        agent_prompts = self._create_agent_prompts(
            issue,
            agent_opinions,
            "",  # Base user prompt needs to be constructed fully for sequence eval
        )

        agent_total_logprobs = []
        valid_scores = True
        logger.debug("        Getting total sequence logprobs for each agent:")
        for agent_idx, (agent_sys_prompt, agent_user_prompt_template) in enumerate(
            agent_prompts
        ):
            # Construct the full user prompt for the agent including the final statement
            # This assumes the template ends where the statement should begin.
            # full_agent_user_prompt = f"{agent_user_prompt_template}{final_statement}"
            logger.debug(
                f"          Agent {agent_idx} evaluating sequence ending with: ...{final_statement[-50:]}"
            )

            system_prompt = (
                f"{agent_sys_prompt}\n\n{agent_user_prompt_template}{current_statement}"
            )
            user_prompt = rollout_sequence

            eval_seed = base_seed + rollout_depth + agent_idx  # Offset seed
            total_logprob = self._get_agent_sequence_logprob(
                agent_system_prompt=system_prompt,
                agent_user_prompt_full=user_prompt,
                seed=eval_seed,
            )

            if total_logprob is None:
                logger.warning(
                    f"          Agent {agent_idx}: Failed to get total logprob for final sequence from agent {agent_idx}."
                )
                valid_scores = False
                break
            agent_total_logprobs.append(total_logprob)
            logger.debug(
                f"          Agent {agent_idx} total logprob: {total_logprob:.4f}"
            )

        if not valid_scores or not agent_total_logprobs:
            logger.warning(
                "      Rollout evaluation failed for one or more agents. Assigning low reward."
            )
            return -100.0  # Assign penalty

        min_reward = min(agent_total_logprobs)
        logger.debug(
            f"      Rollout evaluation complete. Agent Total Logprobs: {[f'{lp:.2f}' for lp in agent_total_logprobs]}, Min Reward: {min_reward:.4f}"
        )
        return min_reward

    def _expand_and_evaluate(
        self, node: Node, issue: str, agent_opinions: dict, step_seed: Optional[int]
    ) -> float:
        """
        Expands the node by trying one untried token.
        Samples tokens if needed, creates a child node.
        If child is terminal, evaluates the token using agent logprobs.
        If child is not terminal, performs a rollout simulation.
        Returns the reward (either immediate token reward or rollout reward).
        """
        logger.debug(f"    Starting Expansion/Evaluation for node: '{node.statement}'")
        reward = -100.0  # Default pessimistic reward

        # 1. Sample potential next tokens if not already done
        if node.untried_tokens is None:
            logger.debug(
                f"      Node needs sampling for potential next tokens (width={self.expansion_sample_width})."
            )
            ref_sys_prompt, ref_user_prompt = self._create_reference_prompt(
                issue, agent_opinions, node.statement
            )
            sampling_seed = (
                step_seed if step_seed is not None else random.randint(0, 10000)
            ) + node.visits  # Vary seed
            node.untried_tokens = self._sample_next_tokens(
                system_prompt=ref_sys_prompt,
                user_prompt=ref_user_prompt,
                num_desired_tokens=self.expansion_sample_width,
                max_attempts=self.max_sampling_attempts,
                temperature=1.0,  # Consider making configurable
                base_seed=sampling_seed,
            )
            node.untried_tokens = [
                t for t in node.untried_tokens if t[0] not in node.children
            ]
            logger.debug(
                f"      Sampled {len(node.untried_tokens)} unique untried tokens: {[t[0] for t in node.untried_tokens]}"
            )
            if not node.untried_tokens:
                logger.warning(
                    f"      Sampling yielded no new untried tokens for node '{node.statement}'."
                )

        # 2. If there are untried tokens, expand one
        if node.untried_tokens:
            # --- Expansion Step ---
            next_token_info = node.untried_tokens.pop(0)  # Take the first untried token
            next_token = next_token_info[0]
            # ref_logprob = next_token_info[1] # Reference logprob is available if needed
            logger.debug(f"      Expanding with untried token: '{next_token}'")

            # 3. Create the new statement and check for terminal state
            new_statement = self._append_to_statement(node.statement, next_token)
            is_terminal = next_token.strip() in self.LLAMA3_EOS_TOKENS
            logger.debug(
                f"        New statement: '{new_statement}' (Terminal: {is_terminal})"
            )

            # 4. Create the child node
            child_node = Node(
                statement=new_statement,
                parent=node,
                token=next_token,
                is_terminal=is_terminal,
            )
            node.children[next_token] = child_node  # Add child here
            logger.debug(
                f"      Created child node for token '{next_token}'. Parent: '{node.statement}', Child: '{child_node.statement}'"
            )

            # --- Evaluation Step (evaluate immediate token reward AND perform rollout if non-terminal) ---
            # 5. First evaluate the immediate reward of the token for all nodes
            logger.debug(
                f"        Evaluating immediate token reward for '{next_token}'."
            )
            agent_prompts = self._create_agent_prompts(
                issue,
                agent_opinions,
                node.statement,  # Context is statement *before* next_token
            )
            agent_logprobs = []
            valid_scores = True
            logger.debug(
                f"          Getting token logprobs for '{next_token}' from {len(agent_prompts)} agents:"
            )
            for agent_idx, (agent_sys_prompt, agent_user_prompt) in enumerate(
                agent_prompts
            ):
                eval_seed = (
                    (step_seed if step_seed is not None else random.randint(0, 10000))
                    + node.visits * len(agent_prompts)  # Use different seed offset
                    + agent_idx
                    + 1  # Add 1 to avoid collision with sampling
                )
                logprob = self._get_agent_token_logprob(
                    agent_system_prompt=agent_sys_prompt,
                    agent_user_prompt_before_token=agent_user_prompt,
                    token_to_evaluate=next_token,
                    seed=eval_seed,
                )
                if logprob is None:
                    logger.warning(
                        f"          Agent {agent_idx}: Failed to get logprob for token '{next_token}'."
                    )
                    valid_scores = False
                    break
                agent_logprobs.append(logprob)
                logger.debug(
                    f"          Agent {agent_idx} logprob for '{next_token}': {logprob:.4f}"
                )

            if valid_scores and agent_logprobs:
                immediate_reward = min(agent_logprobs)
                logger.debug(
                    f"        Token '{next_token}' immediate evaluation complete. Agent Logprobs: {[f'{lp:.2f}' for lp in agent_logprobs]}, Min Reward: {immediate_reward:.4f}"
                )
                # Store immediate reward on the child node for future use
                child_node.immediate_reward = immediate_reward
            else:
                logger.warning(
                    f"        Token evaluation failed. Assigning low immediate reward."
                )
                immediate_reward = -100.0  # Penalty for evaluation failure
                child_node.immediate_reward = immediate_reward

            # Terminal nodes don't need rollout, just use immediate reward
            if child_node.is_terminal:
                reward = immediate_reward
                logger.debug(
                    f"        Node is terminal, using immediate reward: {reward:.4f}"
                )
            else:
                # For non-terminal nodes, perform rollout AND combine with immediate reward
                logger.debug(
                    f"        Child node is not terminal. Performing rollout simulation from '{child_node.statement}'."
                )
                rollout_seed = (
                    (step_seed if step_seed is not None else random.randint(0, 10000))
                    + node.visits * (len(agent_opinions) + 1)  # Different offset
                    + 5000  # Arbitrary large offset
                )
                rollout_reward = self._rollout(
                    start_node=child_node,
                    issue=issue,
                    agent_opinions=agent_opinions,
                    rollout_depth=self.rollout_depth,
                    seed=rollout_seed,
                )
                # Combine immediate reward with rollout reward using gamma hyperparameter
                reward = immediate_reward + self.gamma * rollout_reward
                logger.debug(
                    f"        Combined reward = {immediate_reward:.4f} (immediate) + {self.gamma:.2f}×{rollout_reward:.4f} (rollout) = {reward:.4f}"
                )

            # 6. Return the calculated reward for backpropagation
            logger.debug(
                f"    Expansion/Evaluation finished for token '{next_token}'. Reward to backpropagate: {reward:.4f}"
            )
            return reward

        else:
            # Node has no untried tokens left
            logger.debug(f"      Node '{node.statement}' has no untried tokens left.")
            if (
                not node.children and not node.is_terminal
            ):  # If sampling failed AND it's not already marked terminal
                node.is_terminal = True
                logger.warning(
                    f"      Marking node '{node.statement}' as terminal because sampling yielded no valid children."
                )
                return 0.0  # Return neutral reward as it's now considered terminal
            elif node.is_terminal:
                logger.debug(
                    f"      Node '{node.statement}' is already terminal. No expansion possible."
                )
                return 0.0  # Return neutral reward for terminal node
            else:
                # This case means it's fully expanded (all sampled tokens led to children)
                # but the node itself isn't terminal. The selection phase should handle moving
                # to a child. Returning 0.0 here signifies no *new* reward was generated
                # in this expansion attempt because there was nothing left to expand.
                logger.debug(
                    f"      Node '{node.statement}' is fully expanded but not terminal. No reward generated in this step."
                )
                return 0.0

    def _backpropagate(self, node: Node, reward: float):
        """Backpropagates the reward up the tree from the expanded node."""
        # --- Ensure reward is not None ---
        if reward is None:
            logger.error(
                f"Attempted to backpropagate a None reward from node '{node.statement}'. Using -100.0 instead."
            )
            reward = -100.0
        # --- End Ensure ---

        logger.debug(
            f"    Starting Backpropagation with reward {reward:.4f} from node '{node.statement}' (current visits={node.visits}, value={node.value:.4f})"
        )
        current_node = node
        path = []  # Keep track of the path for logging
        while current_node is not None:
            path.append(current_node.statement)
            current_node.visits += 1
            current_node.total_reward += reward
            # Update average reward (value)
            # Avoid division by zero if visits somehow becomes 0 (shouldn't happen here)
            old_value = current_node.value
            current_node.value = (
                current_node.total_reward / current_node.visits
                if current_node.visits > 0
                else 0.0
            )
            logger.debug(
                f"      Updating node '{current_node.statement}': visits={current_node.visits}, total_reward={current_node.total_reward:.4f}, value={old_value:.4f} -> {current_node.value:.4f}"
            )
            current_node = current_node.parent
        logger.debug(
            f"    Backpropagation finished. Path: {' <- '.join(reversed(path))}"
        )

    def _select_best_child(self, node: Node) -> Optional[Node]:
        """Selects the best child node after simulations, usually the most visited."""
        if not node.children:
            logger.warning(f"Node '{node.statement}' has no children to select from.")
            return None

        # Sort children by visit count (most robust choice)
        sorted_children = sorted(
            node.children.values(), key=lambda c: c.visits, reverse=True
        )

        best_child = sorted_children[0]
        logger.info(  # Changed to info for step summary
            f"    Selected best child to advance: '{best_child.token}' (Visits: {best_child.visits}, Value: {best_child.value:.4f})"
        )

        # Log other top children for debugging
        for i, child in enumerate(sorted_children[1:5]):
            logger.debug(
                f"      Other child {i+2}: '{child.token}' (Visits: {child.visits}, Value: {child.value:.4f})"
            )

        return best_child

    # --- Main Generation Method ---
    def generate_statement(self, issue: str, agent_opinions: dict) -> str:
        """Generates a statement using MCTS logic."""
        logger.info(
            f"Generating statement for '{issue}' using {self.__class__.__name__} (Egalitarian Welfare)"
        )
        logger.debug(f"  Full Config: {self.config}")

        if not client:
            logger.error("Together client not initialized.")
            return "[ERROR: CLIENT NOT INITIALIZED]"

        num_agents = len(agent_opinions)
        if num_agents == 0:
            logger.warning("No agent opinions provided.")
            return ""

        # Initialize MCTS root
        root = Node(statement="")
        current_statement = ""
        start_time = time.time()

        for step in range(self.max_tokens):
            step_start_time = time.time()
            logger.info(f"\n--- MCTS Step {step + 1}/{self.max_tokens} ---")
            logger.info(f"  Current statement: '\033[1;32m{current_statement}\033[0m'")
            logger.debug(
                f"  Current root node: '{root.statement}' (visits={root.visits}, value={root.value:.4f}, terminal={root.is_terminal})"
            )

            if root.is_terminal:
                logger.info("  Root node is terminal. Stopping generation.")
                break

            # Base seed for this step's simulations
            step_seed = (
                self.seed
                + step * self.num_simulations * (num_agents + 1) * 2  # More spacing
                if self.seed is not None
                else None
            )

            # Run MCTS simulations
            for sim in range(self.num_simulations):
                sim_seed = (step_seed + sim) if step_seed is not None else None
                logger.debug(
                    f"\n  Simulation {sim + 1}/{self.num_simulations} starting (Seed: {sim_seed})"
                )

                # 1. Selection
                logger.debug("  Sim: --- Selection Phase ---")
                selected_node = self._select(root)
                logger.debug(
                    f"  Sim: Selected node '{selected_node.statement}' for expansion/evaluation."
                )

                # 2. Expansion & Evaluation (if not terminal)
                reward = 0.0
                node_to_backprop = (
                    selected_node  # Default to backpropagating from the selected node
                )
                if not selected_node.is_terminal:
                    logger.debug("  Sim: --- Expansion & Evaluation Phase ---")
                    # Pass sim_seed for consistent sampling/evaluation within the simulation
                    reward = self._expand_and_evaluate(
                        selected_node, issue, agent_opinions, sim_seed
                    )
                    # Backpropagation should start from the node that was expanded (selected_node).
                    # The reward obtained reflects the value estimated from this expansion point.
                    logger.debug(
                        f"  Sim: Expansion/Evaluation resulted in reward: {reward:.4f}"
                    )

                else:
                    # If selection led to an already terminal node
                    reward = 0.0  # No new reward generated
                    logger.debug(
                        f"  Sim: Selected node '{selected_node.statement}' is already terminal. No expansion. Reward for backprop: {reward}"
                    )

                # 3. Backpropagation
                logger.debug(f"  Sim: --- Backpropagation Phase ---")
                self._backpropagate(node_to_backprop, reward)
                logger.debug(f"  Sim: Simulation {sim + 1} finished.")

            # After all simulations for this step, select the best child to proceed
            logger.info(f"\n  --- Step {step+1} Summary ---")
            logger.info(
                f"  Root node stats after {self.num_simulations} simulations: visits={root.visits}, value={root.value:.4f}"
            )
            if not root.children:
                logger.warning(
                    f"  Root node '{root.statement}' has no children after simulations. Cannot select best child. Stopping."
                )
                break

            best_child = self._select_best_child(root)

            if best_child is None:
                logger.error(
                    "MCTS simulations did not yield a best child. Stopping."
                )  # Changed to error
                break

            # Update current state
            chosen_token = best_child.token
            current_statement = best_child.statement
            root = best_child  # Move root down to the chosen child
            root.parent = None  # Detach from the old tree (memory optimization)

            logger.info(f"  Chose token: '{chosen_token}'")
            logger.info(f"  New statement: '{current_statement}'")
            logger.info(
                f"  Step {step + 1} duration: {time.time() - step_start_time:.2f}s"
            )

            # Check if the chosen token is an EOS token
            if chosen_token and chosen_token.strip() in self.LLAMA3_EOS_TOKENS:
                logger.info("  EOS token chosen. Finalizing statement.")
                root.is_terminal = True  # Mark as terminal
                break  # Stop generation

        total_time = time.time() - start_time
        logger.info(f"\n--- MCTS Generation Complete ---")
        logger.info(f"  Final statement: '{current_statement}'")
        logger.info(f"  Total generation time: {total_time:.2f}s")
        # Optionally log final node stats
        if root:
            logger.info(f"  Final node visits: {root.visits}")
            logger.info(f"  Final node average reward (value): {root.value:.4f}")

        final_statement = current_statement.strip()

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

        return final_statement
