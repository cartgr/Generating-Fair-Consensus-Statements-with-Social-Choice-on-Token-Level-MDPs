from .base import BaseGenerator
from ..utils import generate_text, get_prompt_logprobs, brushup_statement_ending
import numpy as np
import logging  # Import the logging module
from typing import Optional, Tuple
import time  # Import the time module

logger = logging.getLogger(__name__)  # Get logger specific to this module


class FiniteLookaheadGenerator(BaseGenerator):
    """
    Statement generator using Finite Lookahead for a multi-agent tree search approach.

    Builds a tree of potential token sequences, evaluates each path based on agent-specific
    rewards derived from log-probability differences, and selects the path maximizing
    Nash welfare (product of exponentiated rewards).
    """

    # --- Constants ---
    DEFAULT_REWARD = -10.0
    REWARD_CLIP_MIN = -20.0
    REWARD_CLIP_MAX = 20.0
    UTILITY_EPSILON = 1e-9
    LLAMA3_EOS_TOKENS = {"<|eot_id|>", "<|end_of_text|>"}

    DEFAULT_REFERENCE_SYSTEM_PROMPT = """You are generating a consensus statement that represents the views of multiple participants.
Your task is to continue the statement in a way that addresses the issue and considers all participants' opinions. Be concise and keep the statement short (less than 50 tokens) and focused. ONLY WRITE THE STATEMENT AND NOTHING ELSE."""
    DEFAULT_AGENT_SYSTEM_PROMPT_TEMPLATE = """You are generating a statement that represents the views of a single participant.
Your task is to continue the statement in a way that addresses the issue and considers ONLY this participant's opinion. Be concise and keep the statement short (less than 50 tokens) and focused. ONLY WRITE THE STATEMENT AND NOTHING ELSE."""

    DEFAULT_AGENT_USER_PROMPT_TEMPLATE = """Issue:\n{issue}\n\nAgent's opinion:\n{opinion}\n\nStatement reflecting this opinion (less than 50 tokens):\n"""
    DEFAULT_REFERENCE_USER_PROMPT_TEMPLATE = """Issue:\n{issue}\n\nParticipants' opinions:\n{opinions_text}\n\nConsensus statement (less than 50 tokens):\n"""

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
        # This info message will only appear if logger.level <= INFO
        logger.info(f"FiniteLookaheadGenerator logger level set to: {log_level_str}")

        # Add configuration for API delay
        self.api_delay = self.config.get("api_delay", 0.1)  # Default to 0.1 seconds
        logger.info(f"  API call delay set to: {self.api_delay} seconds")

        # Add configuration for brushup option
        self.brushup = self.config.get("brushup", False)
        self.brushup_model = self.config.get(
            "brushup_model", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        )
        if self.brushup:
            logger.info(f"  Statement brushup enabled with model: {self.brushup_model}")

    def generate_statement(self, issue: str, agent_opinions: dict) -> str:
        """
        Generates a statement using Finite Lookahead logic.

        Args:
            issue: The central issue or topic.
            agent_opinions: A dictionary mapping agent identifiers to their opinions.

        Returns:
            A generated consensus statement based on the lookahead search.
        """
        logger.info(
            f"Generating statement for '{issue}' using {self.__class__.__name__}"
        )
        logger.info(f"  Config: {self.config}")

        # Extract configuration parameters
        branching_factor = self.config.get("branching_factor", 2)
        max_depth = self.config.get("max_depth", 3)
        max_tokens = self.config.get("max_tokens", 50)
        beta = self.config.get("beta", 1.0)
        seed = self.config.get("seed")  # seed is Optional[int]

        logger.info(f"  Branching factor: {branching_factor}")
        logger.info(f"  Max depth: {max_depth}")
        logger.info(f"  Max tokens: {max_tokens}")
        logger.info(f"  Beta: {beta}")
        logger.info(f"  Seed: {seed}")

        # --- Format System Prompts ---
        # We do this once here, and pass the formatted prompts down
        formatted_reference_system_prompt = self.DEFAULT_REFERENCE_SYSTEM_PROMPT
        formatted_agent_system_prompts = self.DEFAULT_AGENT_SYSTEM_PROMPT_TEMPLATE

        # Initialize current statement
        current_statement = ""
        token_count = 0

        # Generate statement token by token
        while token_count < max_tokens:

            logger.info(
                f"Current statement length: {token_count}/{max_tokens} tokens. Statement: \033[1;32m{current_statement}\033[0m"
            )
            logger.debug(f"Current statement: '{current_statement}'")

            # Create a properly formatted prompt for continuation
            prompt_for_continuation = self._create_reference_continuation_prompt(
                issue, agent_opinions, current_statement
            )

            # Step 1: Generate tree paths using reference policy
            tree_paths = self._generate_tree_paths(
                issue,
                agent_opinions,
                current_statement,
                branching_factor,
                max_depth,
                seed,
                prompt_for_continuation,
                formatted_reference_system_prompt,  # Pass formatted prompt
            )

            if not tree_paths:
                logger.warning("No valid tree paths generated. Ending generation.")
                break

            # Step 2: Calculate rewards for each agent for each path
            # This is now called unconditionally to ensure context/logprobs are available
            # This function will now also return tokenization info needed later
            next_token = self._get_first_token_of_best_path(
                issue,
                agent_opinions,
                current_statement,
                tree_paths,  # Pass the generated paths
                beta,
                seed,
                formatted_reference_system_prompt,  # Pass formatted prompt
                formatted_agent_system_prompts,  # Pass formatted prompts
            )

            if next_token.strip() in ["DONE"]:
                break
            if next_token in ["\n", "\n\n", ".\n\n"]:
                break

            if not current_statement:
                current_statement = next_token
            else:
                current_statement = self._append_to_statement(
                    current_statement, next_token
                )

            token_count += 1

        final_statement = current_statement.strip()
        pre_brushup_statement = final_statement  # Save the statement before brushup

        # Apply brushup if enabled
        if self.brushup:
            logger.info("Applying statement ending brushup...")
            brushed_statement = brushup_statement_ending(
                final_statement, self.brushup_model
            )
            if brushed_statement != final_statement:
                logger.info("Statement ending was fixed during brushup")
            else:
                logger.info("Statement ending was already well-formed (not modified)")
            final_statement = brushed_statement

        # Add the pre-brushup statement as a property
        self.pre_brushup_statement = pre_brushup_statement

        return final_statement

    def _reconstruct_path_string(self, token_list: list[str]) -> str:
        """
        Reconstructs the full path string from a list of generated tokens/sequences.
        Uses the _append_to_statement logic for consistent concatenation.

        Args:
            token_list: A list of strings, where each string is a token or sequence generated.

        Returns:
            The reconstructed string representation of the path.
        """
        path_str = ""
        for token in token_list:
            # Use the existing append logic which handles spacing/concatenation
            path_str = self._append_to_statement(path_str, token)
        return path_str

    def _create_reference_continuation_prompt(
        self, issue: str, agent_opinions: dict, current_statement: Optional[str] = None
    ) -> str:
        """Create a properly formatted prompt for continuation."""
        opinions_text = "\n\n".join(
            [
                f"Participant {i+1}: {opinion}"
                for i, opinion in enumerate(agent_opinions.values())
            ]
        )

        # Use the template constant
        base_prefix = self.DEFAULT_REFERENCE_USER_PROMPT_TEMPLATE.format(
            issue=issue, opinions_text=opinions_text
        )

        if current_statement:
            base_prefix = self._append_to_statement(base_prefix, current_statement)

        return base_prefix

    def _create_agent_continuation_prompt(
        self, issue: str, agent_opinions: dict
    ) -> list[str]:
        """Create a properly formatted prompt for continuation."""
        return [
            self.DEFAULT_AGENT_USER_PROMPT_TEMPLATE.format(issue=issue, opinion=opinion)
            for opinion in agent_opinions.values()
        ]

    def _append_to_statement(self, current_statement: str, next_sequence: str) -> str:
        return f"{current_statement}{next_sequence}"

    def _generate_tree_paths(
        self,
        issue: str,
        agent_opinions: dict,
        current_statement: str,
        branching_factor: int,
        max_depth: int,
        seed: Optional[int],
        prompt_for_continuation: Optional[str] = None,
        formatted_reference_system_prompt: Optional[str] = None,  # Added argument
    ) -> list[str]:  # Changed return type annotation
        """
        Generates possible continuations of the current statement using a reference policy.
        Builds a tree recursively by generating one token at a time with branching.

        Args:
            issue: The central issue.
            agent_opinions: Dictionary of agent opinions.
            current_statement: The statement generated so far.
            branching_factor: Number of branches at each node.
            max_depth: Maximum depth of the tree (number of tokens to look ahead).
            seed: Optional random seed for generation.
            prompt_for_continuation: Optional pre-formatted prompt.
            formatted_reference_system_prompt: Optional pre-formatted system prompt.

        Returns:
            A list of unique, non-empty paths, where each path is a string.
        """
        logger.info(
            f"Generating tree paths (as strings) with depth {max_depth} and branching factor {branching_factor}"
        )
        logger.debug(
            f"Building search tree for current statement: '{current_statement}'"
        )

        # Use the formatted system prompt passed as argument
        system_prompt = (
            formatted_reference_system_prompt
            or self.DEFAULT_REFERENCE_SYSTEM_PROMPT  # Fallback if not passed (no formatting needed)
        )

        # Use provided prompt if available, otherwise create one
        if prompt_for_continuation:
            base_prompt = prompt_for_continuation
        else:
            base_prompt = self._create_continuation_prompt(
                issue, agent_opinions, current_statement
            )

        # Helper function for recursive tree building
        def build_tree_paths_recursive(
            prompt: str,
            current_token_list: list[str],
            depth: int,
            path_seed: Optional[int],
        ) -> list[str]:
            # Base case: reached maximum depth
            if depth == 0:
                logger.debug(
                    f"Reached maximum depth with token list: {current_token_list}"
                )
                # Return the completed path (list of tokens)
                # Return list containing the token list to match the expected return type
                return [current_token_list]

            all_child_paths = []  # Collect paths (lists of tokens) from recursive calls

            logger.debug(
                f"Building at depth {max_depth - depth + 1}, current token list prefix: {current_token_list}"
            )

            # Generate branching_factor potential next sequences at this level
            for i in range(branching_factor):
                gen_seed: Optional[int] = None
                if path_seed is not None:
                    # Ensure seeds are distinct for each branch and level
                    gen_seed = path_seed + i * (max_depth + 1)  # Make seeds distinct

                logger.debug(
                    f"Generating branch {i+1}/{branching_factor} with seed {gen_seed}"
                )

                # Generate single token
                try:
                    # Use the prompt passed into this level of recursion
                    next_token_sequence = generate_text(
                        model=self.model_identifier,
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        max_tokens=1,
                        temperature=1.0,
                        seed=gen_seed,
                        bias_against_tokens=[
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
                        ],
                        repetition_penalty=1.0,
                        use_chat_completions=False,
                    )
                    if self.api_delay > 0:
                        time.sleep(self.api_delay)
                    logger.debug(
                        f"Raw generated token sequence: '{next_token_sequence}'"
                    )
                except Exception as e:
                    logger.error(
                        f"Error during text generation for branch {i+1}: {e}",
                        exc_info=True,
                    )
                    next_token_sequence = ""

                # Create new token list for recursion
                new_token_list = current_token_list + [next_token_sequence]

                if next_token_sequence in ["\n", "\n\n", ".\n\n", '."\n\n']:
                    logger.debug(
                        f"Termination token detected. Ending path {new_token_list}"
                    )
                    all_child_paths.append(new_token_list)
                    continue

                logger.debug(
                    f"Appended token '{next_token_sequence}' -> new token list: {new_token_list}"
                )

                # Reconstruct the string representation of the path generated *so far in this branch*
                # This is needed to create the correct prompt for the *next* token prediction
                current_branch_path_str = self._reconstruct_path_string(new_token_list)

                # Combine the initial statement context with the path string generated within this tree branch
                full_statement_for_next_prompt = self._append_to_statement(
                    current_statement, current_branch_path_str
                )
                next_level_prompt = self._create_reference_continuation_prompt(
                    issue,
                    agent_opinions,
                    full_statement_for_next_prompt,
                )

                next_level_seed: Optional[int] = None
                if gen_seed is not None:
                    next_level_seed = gen_seed + 1

                logger.debug(
                    f"Recursing to depth {max_depth - depth + 2} with new token list {new_token_list} and prompt ending: ...{next_level_prompt[-50:]}"
                )

                # Collect full paths (lists of tokens) returned from the recursive call
                child_paths = build_tree_paths_recursive(
                    next_level_prompt, new_token_list, depth - 1, next_level_seed
                )

                if logger.isEnabledFor(logging.DEBUG) and child_paths:
                    logger.debug(
                        f"Generated child paths (token lists) at depth {max_depth - depth + 1}: {child_paths}"
                    )

                all_child_paths.extend(child_paths)

            return all_child_paths

        # Start building tree from the root, passing an empty list
        all_paths_token_lists = build_tree_paths_recursive(
            base_prompt, [], max_depth, seed
        )

        # Remove empty lists and duplicates while preserving order
        unique_paths = []
        seen_paths = set()
        for token_list in all_paths_token_lists:
            # Skip empty lists (e.g., if generation failed at the root)
            if not token_list:
                continue
            # Convert list to tuple for hashing to check uniqueness
            path_tuple = tuple(token_list)
            if path_tuple not in seen_paths:
                unique_paths.append(token_list)
                seen_paths.add(path_tuple)

        logger.info(
            f"Generated {len(unique_paths)} unique tree paths (strings) of depth {max_depth}"
        )
        if logger.isEnabledFor(logging.DEBUG):
            for i, p_str in enumerate(unique_paths):
                logger.debug(f"  Unique Path (String) {i+1}: '{p_str}'")

        return unique_paths  # Return list of strings now

    def _get_first_token_of_best_path(
        self,
        issue: str,
        agent_opinions: dict,
        current_statement: str,
        tree_paths: list[str],
        beta: float,
        seed: Optional[int],
        formatted_reference_system_prompt: str,  # Added argument
        formatted_agent_system_prompts: str,  # Added argument
    ) -> str:  # Return a dict containing rewards and token info
        """
        Calculate rewards for each agent for each potential path continuation.

        Reward is based on the scaled difference in log probability of the path tokens
        between the agent's policy and the reference policy:
          R = beta * sum(log P_agent(token_i | context, path_<i) - log P_ref(token_i | context, path_<i))
        Log probabilities are derived by comparing logprobs of the full text (context + path)
        with the logprobs of the context alone.

        Args:
            issue: The central issue.
            agent_opinions: Dictionary mapping agent IDs to opinions.
            current_statement: The statement generated so far.
            tree_paths: List of path continuations generated by the reference policy.
            beta: Scaling factor for the reward.
            seed: Optional random seed for logprob calculation consistency.

        Returns:
            A dictionary containing:
            - 'agent_rewards': Dict mapping agent IDs to lists of rewards.
            - 'ref_context_tokens': List of tokens for the reference context.
            - 'reference_policy_logprobs_map': Dict mapping path index to (full_tokens, full_logprobs) for reference policy.
        """

        logger.info(f"potential paths: {tree_paths}")

        # path rewards. This should be have a path to agent reward mapping
        path_rewards = {i: [] for i in range(len(tree_paths))}

        for i, path in enumerate(tree_paths):

            reconstructed_path = self._reconstruct_path_string(path)

            # for this given path we want to make the reference policy prompt and the agent policy prompts
            reference_policy_prompt = self._create_reference_continuation_prompt(
                issue, agent_opinions
            )

            potential_statement = self._append_to_statement(
                current_statement, reconstructed_path
            )

            # add the reconstructed path to the reference policy prompt
            reference_policy_prompt = self._append_to_statement(
                reference_policy_prompt, potential_statement
            )

            # reference_policy_logprobs = get_prompt_logprobs(
            #     model=self.model_identifier,
            #     system_prompt=formatted_reference_system_prompt,
            #     user_prompt=reference_policy_prompt,
            # )

            # reference_policy_path_logprobs = reference_policy_logprobs[1][-len(path) :]

            agent_policy_prompts = self._create_agent_continuation_prompt(
                issue, agent_opinions
            )

            agent_rewards = []

            for agent_prompt in agent_policy_prompts:

                agent_prompt = self._append_to_statement(
                    agent_prompt, potential_statement
                )

                agent_policy_logprobs = get_prompt_logprobs(
                    model=self.model_identifier,
                    system_prompt=formatted_agent_system_prompts,
                    user_prompt=agent_prompt,
                )

                agent_policy_path_logprobs = agent_policy_logprobs[1][-len(path) :]

                # now element-wise subtract the reference policy path logprobs from the agent policy path logprobs
                # and multiply by beta
                # agent_reward = beta * (
                #     np.array(agent_policy_path_logprobs)
                #     - np.array(reference_policy_path_logprobs)
                # )

                # # sum the rewards for each agent
                # agent_path_reward = np.sum(agent_reward)

                agent_path_reward = np.mean(np.array(agent_policy_path_logprobs))

                agent_rewards.append(agent_path_reward)

            path_rewards[i] = agent_rewards

        # now pick the path with the highest minimum reward
        best_path_index = max(path_rewards, key=lambda x: min(path_rewards[x]))
        best_path_rewards = path_rewards[best_path_index]
        best_path = tree_paths[best_path_index]
        first_token_of_best_path = best_path[0]

        logger.info(f"best_path_rewards: {best_path_rewards}")
        logger.info(f"best_path:         {best_path}")
        logger.info(f"first_token_of_best_path: {first_token_of_best_path}")

        return first_token_of_best_path
