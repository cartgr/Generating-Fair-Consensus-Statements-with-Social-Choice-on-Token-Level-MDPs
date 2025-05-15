from .base import BaseGenerator
from ..utils import generate_text
import re  # Added import for regex
import numpy as np  # Add numpy import
import random  # Add random import for tie-breaking


class HabermasMachineGenerator(BaseGenerator):
    """
    Statement generator using a custom implementation inspired by the Habermas Machine.
    """

    def generate_statement(self, issue: str, agent_opinions: dict) -> str:
        """Mediates caucus deliberation among participants.

        The Habermas Machine facilitates AI-mediated deliberation among a group of
        participants on a given question. It acts as a "mediator," iteratively
        refining a group statement that aims to capture the common ground of the
        participants' opinions.

        The process involves:

        1. Gathering initial opinions from participants.
        2. Generating candidate group statements using a Large Language Model (LLM).
        3. Evaluating these statements using a personalized reward model, predicting
           the order of preference of each participant for each statement.
        4. Aggregating individual preferences using a social choice method to select
           a winning statement.
        5. Gathering critiques of the winning statement from participants.
        6. Generating revised statements based on the critiques and previous opinions.
        7. Optionally, repeating steps 3-6 for multiple rounds, refining the statement
           iteratively. In the paper, we use one opinion and one critique round.

        This class manages the entire deliberation process, including interaction with
        the LLM, the reward model, and the social choice mechanism.  It maintains the
        history of opinions, critiques, candidate statements, and winning statements
        for each round."""

        print(f"Generating statement for '{issue}' using {self.__class__.__name__}")
        print(f"  Config: {self.config}")
        print(
            f"  Model: {self.model_identifier}"
        )  # Access model identifier from BaseGenerator

        # Access the seed passed down via the config
        current_seed = self.config.get("seed")

        print(f"  Seed for this run: {current_seed}")

        num_candidates = self.config.get("num_candidates", 3)  # Default to 3 if not set
        num_retries = self.config.get("num_retries_on_error", 1)  # Configurable retries
        num_rounds = self.config.get(
            "num_rounds", 1
        )  # Number of critique/revision rounds
        print(f"  Number of candidates to generate: {num_candidates}")
        print(f"  Number of retries on error: {num_retries}")
        print(f"  Number of critique/revision rounds: {num_rounds}")

        # step 1: gather initial opinions from participants

        # step 2: generate candidate statements

        candidate_statements = _generate_candidate_statements(
            issue=issue,
            agent_opinions=agent_opinions,
            num_candidates=num_candidates,
            seed=current_seed,
            model_identifier=self.model_identifier,
        )
        print(f"  Generated {len(candidate_statements)} candidate statements.")
        if not candidate_statements:
            print("Error: Failed to generate any candidate statements.")
            # Return an error message or handle appropriately
            return f"Failed to generate any candidate statements for '{issue}'."
        if len(candidate_statements) == 1:
            print(
                "Warning: Only one candidate statement generated. Skipping ranking and returning the single statement."
            )
            # If only one statement, ranking isn't meaningful. Return it directly?
            # Or proceed with dummy ranking for consistency? Let's return it for now.
            # TODO: Decide final behavior for single candidate
            return candidate_statements[0]

        # step 3: evaluate statements using personalized reward model (predict agent rankings)
        agent_rankings = {}
        agent_explanations = {}
        print(
            f"\nStep 3: Evaluating {len(candidate_statements)} statements for {len(agent_opinions)} agents..."
        )
        # Offset seed for evaluation phase to avoid reusing generation seeds
        current_eval_seed = (
            (current_seed + num_candidates)
            if current_seed is not None
            else np.random.randint(10000)
        )  # Use offset or random if None

        for agent_id, opinion in agent_opinions.items():
            print(f"  Predicting ranking for Agent {agent_id}...")
            ranking, explanation_or_error = _hm_predict_ranking_for_agent(
                model_identifier=self.model_identifier,
                question=issue,
                opinion=opinion,
                statements=candidate_statements,
                seed=current_eval_seed,
                num_retries_on_error=num_retries,  # Use configured retries
            )
            agent_rankings[agent_id] = ranking
            agent_explanations[agent_id] = (
                explanation_or_error  # Store explanation/error
            )

            if ranking is None:
                print(
                    f"    Warning: Failed to get valid ranking for Agent {agent_id}. Error: {explanation_or_error}"
                )
                # Decide how to handle failed ranking: exclude agent, use default, etc.
                # For now, store None. Aggregation step will need to handle this.
            else:
                # Convert numpy array ranking to list for easier display/storage if needed
                ranking_list = ranking.tolist()
                print(
                    f"    Successfully predicted ranking for Agent {agent_id}: {ranking_list}"
                )
                # Optionally print explanation for debugging:
                # print(f"      Explanation snippet: {explanation_or_error[:100]}...")

            # Increment seed for diversity in evaluation per agent, only if it's not None
            if current_eval_seed is not None:
                # Check if seed is actually numeric before incrementing
                if isinstance(current_eval_seed, (int, float)):
                    current_eval_seed += 1
                else:  # Handle case where seed might become non-numeric unexpectedly
                    print("Warning: Seed became non-numeric, cannot increment.")
                    current_eval_seed = np.random.randint(10000)  # Reset to random

        # --- Store results for potential later use ---
        # Store these on the instance if needed by other steps or for analysis
        self.candidate_statements = candidate_statements
        self.agent_rankings = agent_rankings
        self.agent_explanations = agent_explanations
        print("\nStep 3 Evaluation Complete.")
        # Filter out None rankings before printing for clarity
        valid_rankings = {
            k: v.tolist() for k, v in agent_rankings.items() if v is not None
        }
        print(f"  Valid Agent Rankings: {valid_rankings}")
        # --------------------------------------------

        # step 4: aggregate preferences using Schulze method
        print("\nStep 4: Aggregate Preferences using Schulze Method")
        if not candidate_statements:
            # Should have been caught earlier, but double-check
            return "Error: No candidate statements to aggregate."
        if len(candidate_statements) == 1:
            # No aggregation needed for a single candidate
            print("  Only one candidate, selecting it directly.")
            winning_statement = candidate_statements[0]
            winning_statement_index = 0
        else:
            # Use the Schulze implementation with random tie-breaking
            # Pass the seed for deterministic tie-breaking
            social_ranking = _aggregate_schulze(
                agent_rankings,
                len(candidate_statements),
                seed=current_seed,  # Pass the seed
                tie_breaking_method="random",  # Use random tie-breaking
            )

            if social_ranking is None:
                print(
                    "  Error: Schulze aggregation failed (e.g., no valid rankings). Falling back to first statement."
                )
                winning_statement_index = 0
                winning_statement = candidate_statements[0]
            else:
                # Since we use random tie-breaking, there should be a unique winner (rank 0)
                print(f"  Schulze Social Ranking (lower is better): {social_ranking}")
                winners = np.where(social_ranking == 0)[0]  # Winner has rank 0
                if len(winners) == 1:
                    winning_statement_index = winners[0]
                    print(
                        f"  Winning Candidate Index (after tie-breaking if any): {winning_statement_index}"
                    )
                else:
                    # This case should ideally not happen with random tie-breaking,
                    # but handle defensively.
                    print(
                        f"  Warning: Multiple winners found ({winners}) even after tie-breaking. Selecting first winner: Index {winners[0]}."
                    )
                    winning_statement_index = winners[0]

                winning_statement = candidate_statements[winning_statement_index]

        # Rename the initial winning statement to avoid conflict in the loop
        current_winning_statement = winning_statement
        self.initial_winning_statement = (
            current_winning_statement  # Store initial winner if needed
        )

        # --- Store intermediate results per round ---
        self.all_round_data = []

        # --- Start Multi-Round Loop ---
        for round_num in range(num_rounds):
            print(
                f"\n--- Starting Critique/Revision Round {round_num + 1}/{num_rounds} ---"
            )
            round_data = {}  # Store data for this round

            # Ensure we have a valid statement to critique
            if not current_winning_statement:
                print(
                    f"  Cannot proceed with round {round_num + 1}: No valid winning statement from previous step."
                )
                break  # Exit loop if there's no statement

            # step 5: gather critiques (using current_winning_statement)
            print(f"\nStep 5 (Round {round_num + 1}): Gather Critiques")
            agent_critiques = {}
            # Offset seed for critique phase - ensure it increments across rounds
            # Use a base offset and add round number * agents
            critique_base_seed = (
                (
                    current_eval_seed + len(agent_opinions)
                )  # Offset from initial eval seed
                if current_eval_seed is not None
                and isinstance(current_eval_seed, (int, float))
                else np.random.randint(
                    20000, 30000
                )  # Use a different random range if needed
            )
            current_critique_seed = (
                critique_base_seed + round_num * len(agent_opinions)
                if critique_base_seed is not None
                else None
            )

            for agent_id, opinion in agent_opinions.items():
                print(f"  Generating critique for Agent {agent_id}...")
                critique = _generate_critique_for_agent(
                    model_identifier=self.model_identifier,
                    issue=issue,
                    opinion=opinion,
                    winning_statement=current_winning_statement,  # Use current winner
                    seed=current_critique_seed,
                )
                agent_critiques[agent_id] = critique
                if critique:
                    print(f"    Critique generated for Agent {agent_id}.")
                else:
                    print(
                        f"    Warning: Failed to generate critique for Agent {agent_id}."
                    )

                # Increment seed for diversity in critique generation per agent
                if current_critique_seed is not None:
                    if isinstance(current_critique_seed, (int, float)):
                        current_critique_seed += 1
                    else:
                        print(
                            "Warning: Critique seed became non-numeric, cannot increment."
                        )
                        current_critique_seed = np.random.randint(
                            20000, 30000
                        )  # Reset to random

            round_data["agent_critiques"] = agent_critiques
            print(
                f"  Agent Critiques gathered: { {k: (v[:50] + '...' if v else None) for k, v in agent_critiques.items()} }"
            )

            # step 6: generate revised statements
            print(
                f"\nStep 6 (Round {round_num + 1}): Generate Multiple Revised Statements"
            )

            # Offset seed for revised statement generation - ensure it increments across rounds
            revision_base_seed = (
                (
                    critique_base_seed + len(agent_opinions) * (num_rounds + 1)
                )  # Offset from critique seeds
                if critique_base_seed is not None
                and isinstance(critique_base_seed, (int, float))
                else np.random.randint(30000, 40000)  # Use a different random range
            )
            current_revision_seed = (
                revision_base_seed + round_num * num_candidates
                if revision_base_seed is not None
                else None
            )

            revised_statements = _generate_multiple_revised_statements(
                model_identifier=self.model_identifier,
                issue=issue,
                agent_opinions=agent_opinions,
                winning_statement=current_winning_statement,  # Use current winner
                agent_critiques=agent_critiques,
                num_candidates=min(num_candidates, 4),
                seed=current_revision_seed,
                num_retries=num_retries,
            )

            # Skip further steps in this round if we couldn't generate revised statements
            if not revised_statements:
                print(
                    f"  Failed to generate revised statements in round {round_num + 1}. Stopping revisions."
                )
                # Keep the current_winning_statement as the final result from the previous round/initial step
                break  # Exit the loop

            round_data["revised_statements"] = revised_statements

            # step 7: evaluate revised statements using personalized reward model
            print(f"\nStep 7 (Round {round_num + 1}): Evaluating Revised Statements...")
            revised_agent_rankings = {}
            revised_agent_explanations = {}

            # Offset seed for evaluation of revised statements - ensure it increments across rounds
            revised_eval_base_seed = (
                (
                    revision_base_seed + num_candidates * (num_rounds + 1)
                )  # Offset from revision seeds
                if revision_base_seed is not None
                and isinstance(revision_base_seed, (int, float))
                else np.random.randint(40000, 50000)
            )
            current_revised_eval_seed = (
                revised_eval_base_seed + round_num * len(agent_opinions)
                if revised_eval_base_seed is not None
                else None
            )

            for agent_id, opinion in agent_opinions.items():
                print(f"  Predicting ranking for Agent {agent_id}...")
                ranking, explanation_or_error = _hm_predict_ranking_for_agent(
                    model_identifier=self.model_identifier,
                    question=issue,
                    opinion=opinion,
                    statements=revised_statements,
                    seed=current_revised_eval_seed,
                    num_retries_on_error=num_retries,
                )
                revised_agent_rankings[agent_id] = ranking
                revised_agent_explanations[agent_id] = explanation_or_error

                if ranking is None:
                    print(
                        f"    Warning: Failed to get valid ranking for Agent {agent_id}. Error: {explanation_or_error}"
                    )
                else:
                    ranking_list = ranking.tolist()
                    print(
                        f"    Successfully predicted ranking for Agent {agent_id}: {ranking_list}"
                    )

                # Increment seed for diversity in evaluation per agent
                if current_revised_eval_seed is not None and isinstance(
                    current_revised_eval_seed, (int, float)
                ):
                    current_revised_eval_seed += 1
                else:
                    print(
                        "Warning: Evaluation seed became non-numeric, cannot increment."
                    )
                    current_revised_eval_seed = np.random.randint(40000, 50000)

            round_data["revised_agent_rankings"] = revised_agent_rankings
            round_data["revised_agent_explanations"] = revised_agent_explanations

            # step 8: aggregate revised preferences using Schulze method
            print(
                f"\nStep 8 (Round {round_num + 1}): Aggregate Revised Preferences using Schulze Method"
            )
            next_winning_statement = (
                None  # Initialize variable for the winner of this round
            )
            if len(revised_statements) == 1:
                print("  Only one revised statement, selecting it directly.")
                next_winning_statement = revised_statements[0]
                # final_statement_index = 0 # Index not strictly needed here
            else:
                # Use Schulze with random tie-breaking
                # Use a seed derived from the revision seed for this round's aggregation
                aggregation_seed = (
                    current_revision_seed
                    if current_revision_seed is not None
                    else np.random.randint(50000, 60000)
                )
                revised_social_ranking = _aggregate_schulze(
                    revised_agent_rankings,
                    len(revised_statements),
                    seed=aggregation_seed,
                    tie_breaking_method="random",
                )

                if revised_social_ranking is None:
                    print(
                        f"  Error: Schulze aggregation for revised statements failed in round {round_num + 1}. Keeping previous winning statement."
                    )
                    # Keep current_winning_statement as is, loop will continue or end
                    next_winning_statement = (
                        current_winning_statement  # Fallback explicitly
                    )
                else:
                    print(
                        f"  Schulze Social Ranking for Revised Statements: {revised_social_ranking}"
                    )
                    winners = np.where(revised_social_ranking == 0)[0]
                    if len(winners) == 1:
                        final_statement_index = winners[0]
                        print(
                            f"  Winning Revised Statement Index: {final_statement_index}"
                        )
                    else:
                        print(
                            f"  Multiple winners found: {winners}. Selecting first winner: Index {winners[0]}."
                        )
                        final_statement_index = winners[0]

                    next_winning_statement = revised_statements[final_statement_index]

            # Update the current winning statement for the next iteration or final result
            current_winning_statement = next_winning_statement
            round_data["winning_statement"] = current_winning_statement
            self.all_round_data.append(round_data)  # Store this round's data
            print(
                f"--- Round {round_num + 1} Complete. Current winning statement: {current_winning_statement[:100]}... ---"
            )
            # --- End Multi-Round Loop ---

        # The final statement is the winning statement after the last successful round
        final_statement = current_winning_statement

        print(
            f"\nFinal consensus statement after {num_rounds} round(s):\n{final_statement}"
        )
        return final_statement


def _generate_initial_prompt(issue: str, opinions: list[str]) -> str:
    """Generates a prompt for the LLM using only the opinions."""
    prompt = f"""
You are assisting a citizens' jury in forming an initial consensus opinion on an important question. The jury members have provided their individual opinions. Your role is to generate a draft consensus statement that captures the main points of agreement and represents the collective view of the jury. The draft statement must not conflict with any of the individual opinions.

Please think through this task step-by-step:

1. Carefully analyze the individual opinions, noting key themes, points of agreement, and areas of disagreement.
2. Based on the analysis, synthesize a concise and clear consensus statement that represents the shared perspective of the jury members. Address the core issue posed in the question, and ensure the statement *does not conflict* with any of the individual opinions. Refer to specific opinion numbers to demonstrate how the draft reflects the collective view.
3. Keep the statement to less than 50 tokens.

Provide your answer in the following format:
<answer>
[Your step-by-step reasoning and explanation for the statement]
<sep>
[Draft consensus statement]
</answer>

Example:
<answer>
1. Most opinions emphasize the importance of public transportation (Opinions 1, 3, 4) and reducing car dependency (Opinions 2, 4). Some also mention cycling and walking as important additions (Opinions 2, 3).
2. The draft statement prioritizes investment in public transport and encourages cycling and walking, reflecting the shared views expressed in the majority of opinions.
<sep>
We believe that investing in public transport, along with promoting cycling and walking, are crucial steps towards creating a more sustainable and livable city.
</answer>


Below you will find the question and the individual opinions of the jury members.

Question: {issue}

Individual Opinions:
"""

    for i, opinion in enumerate(opinions):
        prompt += f"Opinion Person {i+1}: {opinion}\n"

    return prompt.strip()


def _process_llm_response(response: str) -> str | None:
    """Processes the model's response, extracting the statement.
    Handles cases where the terminator might remove the closing tag.

    Args:
        response: The raw model response.

    Returns:
        The extracted statement string, or None if the response format is incorrect.
    """
    # Regex Explanation:
    # <answer> - Match the opening tag (case-insensitive)
    # (.*?)    - Capture group 1: Reasoning (non-greedy, multiline)
    # <sep>    - Match the separator tag (case-insensitive)
    # (.*?)    - Capture group 2: Statement (non-greedy, multiline)
    # (?:</answer>|\Z) - Match either the closing tag OR the absolute end of the string (\Z)
    #                   This handles cases where the terminator cuts off </answer>.
    #                   (?:...) is a non-capturing group.
    match = re.search(
        r"<answer>(.*?)<sep>(.*?)(?:</answer>|\Z)",
        response,
        re.DOTALL | re.IGNORECASE,  # DOTALL allows '.' to match newlines
    )

    statement = None
    if match:
        # Group 2 is the statement after <sep>
        statement = match.group(2).strip()
        # Optional: print reasoning for debugging: print(f"DEBUG Reasoning: {match.group(1).strip()}")
    else:
        # If the regex fails, the format is genuinely unexpected.
        print(
            f"Warning: Could not parse response format. Response:\n{response[:500]}..."
        )
        return None  # Exit early if match fails

    # Basic validation
    if statement and len(statement) > 5:
        return statement
    else:
        # Handle cases where regex matched but statement is missing/too short
        if not statement:
            print(
                f"Warning: Regex matched but statement extraction resulted in empty string. Response:\n{response[:500]}..."
            )
        elif len(statement) <= 5:  # Statement was too short
            print(f"Warning: Extracted statement seems too short: '{statement}'")
        return None


def _generate_candidate_statements(
    issue: str,
    agent_opinions: dict,
    num_candidates: int,
    seed: int | None,
    model_identifier: str,
) -> list[str]:
    """Generate candidate statements using the LLM."""

    opinions_list = list(agent_opinions.values())
    prompt = _generate_initial_prompt(issue, opinions_list)

    candidate_statements = []
    current_seed = seed

    print(f"  Generating {num_candidates} candidates using seed {current_seed}...")

    for i in range(num_candidates):
        print(f"    Generating candidate {i+1}/{num_candidates}...")
        response = generate_text(
            model=model_identifier,
            system_prompt="",
            user_prompt=prompt,
            seed=current_seed,
        )

        # --- Modify this block to print the full response ---
        print("-" * 20 + f" Raw Response Candidate {i+1} " + "-" * 20)
        # Print the entire response string without slicing
        print(response)
        print("-" * (42 + len(str(i + 1))))  # Adjust divider length
        # -----------------------------------------------------

        statement = _process_llm_response(response)

        if statement:
            candidate_statements.append(statement)
            print(f"      Successfully generated candidate {i+1}.")
        else:
            # The warning in _process_llm_response might still truncate,
            # but the print above will show the full context.
            print(f"      Failed to generate or parse candidate {i+1}. Skipping.")

        # Increment seed for diversity in generation, only if it's not None
        if current_seed is not None:
            current_seed += 1

    # Add a check if fewer candidates were generated than requested
    if len(candidate_statements) < num_candidates:
        print(
            f"Warning: Only generated {len(candidate_statements)} valid candidates out of {num_candidates} requested."
        )

    return candidate_statements


def _hm_generate_opinion_only_ranking_prompt(
    question: str,
    opinion: str,
    statements: list[str],
) -> str:
    """Generates a prompt for the LLM to rank statements based only on the opinion."""
    # This prompt is copied directly from _generate_opinion_only_prompt
    # in habermas_machine/reward_model/cot_ranking_model.py
    prompt = f"""
Task: As an AI assistant, your job is to rank these statements in the order that the participant would most likely agree with them, based on their opinion. Use Arrow notation for the ranking, where ">" means "preferred to". Ties are NOT allowed and items should be in descending order of preference so you can ONLY use ">" and the letters of the statements in the final ranking. Examples of valid final rankings: B > A, D > A > C > B. B > C > A > E > D.

Please think through this task step-by-step:

1. Analyze the participant's opinion, noting key points and sentiments.
2. Compare each statement to the participant's opinion, considering how well it aligns with or supports their view.
3. Consider any nuances or implications in the statements that might appeal to or repel the participant based on their expressed opinion.
4. Rank the statements accordingly using only ">" and the letters of the statements.

Provide your answer in the following format:
<answer>
[Your step-by-step reasoning and explanation for the ranking]
<sep>
[Final ranking using arrow notation]
</answer>

For example for five statements A, B, C, D and E a valid response could be:
<answer>
1. The participant's opinion emphasizes the importance of environmental protection and the need for immediate action to address climate change.

2. - Statement A directly addresses the urgency of climate action and proposes concrete steps, aligning with the participant's opinion.
   - Statements B and D acknowledge the seriousness of climate change but offer less concrete solutions. B focuses on global cooperation, while D emphasizes economic considerations.
   - Statement C downplays the urgency of climate change, contradicting the participant's stance.
   - Statement E completely opposes the participant's view by denying the existence of climate change.

3.  The participant's emphasis on immediate action suggests a preference for proactive solutions and a dislike for approaches that downplay the issue or offer only abstract ideas.

4. Based on this analysis, the ranking is: A > D > B > C > E

<sep>
A > D > B > C > E
</answer>

It is important to follow the template EXACTLY. So ALWAYS start with <answer>, then the explanation, then <sep> then only the final ranking and then </answer>.


Below you will find the question and the participant's opinion. You will also find a list of statements to rank.

Question: {question}

Participant's Opinion: {opinion}

Statements to rank:
"""
    for i, statement in enumerate(statements):
        letter = chr(ord("A") + i)  # A, B, C, D, etc.
        # Basic cleaning similar to the reference code
        try:
            cleaned_statement = (
                statement.strip().strip('"').strip("'").strip("\n").strip()
            )
        except Exception as e:
            print(f"Warning: Could not clean statement {i}: {statement}. Error: {e}")
            cleaned_statement = statement  # Use original if cleaning fails
        prompt += f"{letter}. {cleaned_statement}\n"

    # Ensure the prompt ends correctly before the LLM call
    prompt += "\nProvide your answer:"

    return prompt.strip()


def _hm_check_response_format(response: str) -> bool:
    """
    Checks if the response is in the expected <answer>...<sep>...</answer> format.
    NOTE: Reverted to strict check based on test_hm_process_ranking_response behavior.
    """
    # Check for the presence of all three tags, case-insensitive
    has_answer_open = re.search(r"<answer>", response, re.IGNORECASE) is not None
    has_sep = re.search(r"<sep>", response, re.IGNORECASE) is not None
    has_answer_close = re.search(r"</answer>", response, re.IGNORECASE) is not None
    return has_answer_open and has_sep and has_answer_close  # Strict check


def _hm_check_arrow_format(ranking_str: str, num_statements: int) -> bool:
    """
    Checks if the ranking string has a valid arrow/equality format (e.g., 'A>B=C')
    and contains the correct letters for the given number of statements.
    - Allows '>' and '=' separators.
    - Checks for correct letters (A, B, C...) based on num_statements.
    - Checks for duplicate letters.
    """
    if not ranking_str:
        return False

    # Allow '>' or '=' between letters, with optional spaces
    # Pattern: Start with a letter, followed by zero or more groups of [> or =] and a letter.
    pattern = r"^[A-Z](?: *[>=] *[A-Z])*$"
    if not re.fullmatch(pattern, ranking_str):
        print(f"Debug CheckArrow: Failed basic format check: '{ranking_str}'")
        return False

    # Extract letters
    letters = "".join(filter(str.isalpha, ranking_str))

    # Check if the set of letters matches the expected letters
    expected_letters = set(chr(ord("A") + i) for i in range(num_statements))
    actual_letters = set(letters)

    if actual_letters != expected_letters:
        print(
            f"Debug CheckArrow: Letter mismatch. Expected: {expected_letters}, Got: {actual_letters} in '{ranking_str}'"
        )
        return False

    # Check for duplicate letters *within the extracted string* (e.g. A>A is invalid even for num_statements=1)
    # Note: The original absl test didn't explicitly check this *within* the string, only that the set matched num_statements.
    # Let's keep the stricter check for duplicates for robustness.
    if len(letters) != len(actual_letters):
        print(f"Debug CheckArrow: Duplicate letters found in '{ranking_str}'")
        # This check might be stricter than the original if the original allowed A>A for num_statements=1,
        # but A>A is generally not a useful ranking. Let's keep this check.
        # However, the original tests *did* list A>B>A as incorrect, implying duplicates are bad.
        return False

    # Check if the number of elements implied by the letters matches num_statements
    # This is implicitly covered by the expected_letters check above.

    return True


def _hm_extract_arrow_ranking(text: str) -> str | None:
    """
    Extracts the first arrow/equality ranking string found in the text.
    Mimics the original re.search behavior. Cleans internal spaces.
    Uses word boundaries to avoid matching single letters within words.
    Example: ' A > B < C' -> 'A>B'
    Example: ' A = B > C ' -> 'A=B>C'
    Example: ' A ' -> 'A'
    Example: 'Explanation' -> None
    """
    if not text:
        return None

    # Pattern using word boundaries (\b) to find standalone sequences
    # like A, A>B, A=B, A > B = C etc.
    # It looks for a letter at a word boundary, followed by zero or more groups of [separator + letter]
    # and ending at a word boundary.
    pattern = r"\b[A-Z](?: *[>=] *[A-Z])*\b"  # Added word boundaries
    match = re.search(pattern, text)

    if match:
        ranking_str = match.group(0)
        # Clean up spaces *within* the matched group for consistency
        cleaned_ranking = re.sub(r" *([>=]) *", r"\1", ranking_str)
        # Also strip leading/trailing whitespace from the whole match
        cleaned_ranking = cleaned_ranking.strip()
        print(
            f"Debug ExtractArrow: Found match: '{ranking_str}', cleaned: '{cleaned_ranking}'"
        )
        return cleaned_ranking
    # Removed the separate single-letter check as \b[A-Z]\b handles it.

    print(f"Debug ExtractArrow: No ranking pattern found in: '{text[:50]}...'")
    return None


def _parse_arrow_ranking_to_array(
    arrow_ranking: str, num_statements: int
) -> np.ndarray | None:
    """
    Parses a validated arrow/equality ranking string into a numpy rank array.
    Handles ties correctly (assigns the same rank, increments rank by 1 for next level).
    Assumes arrow_ranking has already been validated by _hm_check_arrow_format.
    Rank 0 is the highest preference.
    Example: "B>A=D>C", 4 -> [1 0 2 1] (B=0, A=1, D=1, C=2)
    Example: "A=B=C=D", 4 -> [0 0 0 0]
    Example: "A", 1 -> [0]
    """
    if not arrow_ranking:
        return None

    ranking_arr = np.full(
        num_statements, -1, dtype=int
    )  # Initialize with -1 (unranked)
    current_rank = 0
    processed_letters = set()

    # Split by '>' to handle groups of potentially tied items
    preference_groups = arrow_ranking.split(">")

    for group in preference_groups:
        # Clean whitespace within the group just in case
        group = group.strip()
        if not group:
            continue  # Skip empty groups if '>>' occurred (should be caught by validation)

        tied_items = group.split("=")
        group_letters = set()
        # num_in_group = 0 # No longer needed for rank calculation

        for item in tied_items:
            letter = item.strip()
            # Basic validation (should be redundant if pre-validated)
            if not ("A" <= letter <= "Z" and len(letter) == 1):
                print(f"Error parsing group '{group}': Invalid item '{letter}'")
                return None
            if letter in processed_letters:
                print(
                    f"Error parsing group '{group}': Duplicate letter '{letter}' across groups"
                )
                return None
            idx = ord(letter) - ord("A")
            if not (0 <= idx < num_statements):
                print(
                    f"Error parsing group '{group}': Letter '{letter}' out of bounds for {num_statements} statements"
                )
                return None

            # Assign current rank to this letter
            ranking_arr[idx] = current_rank
            group_letters.add(letter)
            # num_in_group += 1 # No longer needed

        processed_letters.update(group_letters)
        # Increment rank by 1 for the next preference level, regardless of group size
        current_rank += 1

    # Final validation: Check if all statements were ranked
    # This ensures _hm_check_arrow_format logic aligns with parser logic
    expected_letters = set(chr(ord("A") + i) for i in range(num_statements))
    if processed_letters != expected_letters:
        missing = expected_letters - processed_letters
        extra = processed_letters - expected_letters
        print(
            f"Error parsing ranking '{arrow_ranking}': Letter set mismatch. Missing: {missing}, Extra: {extra}. Check validation."
        )
        # This should ideally be caught by _hm_check_arrow_format beforehand
        return None  # Fail if set of letters doesn't match num_statements

    # Check if any rank remained -1 (shouldn't happen if letter sets match)
    if -1 in ranking_arr:
        print(
            f"Error parsing ranking '{arrow_ranking}': Not all statements were assigned a rank. Result: {ranking_arr}"
        )
        return None

    return ranking_arr


def _hm_process_ranking_response(
    response: str, num_statements: int
) -> tuple[np.ndarray | None, str]:
    """
    Processes the LLM response string to extract ranking and explanation.
    Mimics the original cot_ranking_model._process_model_response logic.
    Handles ties and uses the original error messages/explanation formats.
    """
    ranking_arr = None
    explanation = (
        response  # Default explanation is the full response for success/fallback
    )

    # 1. Check standard format <answer>...<sep>...</answer>
    is_correct_format = _hm_check_response_format(response)

    if is_correct_format:
        # Extract text between <sep> and </answer>
        sep_match = re.search(r"<sep>", response, re.IGNORECASE)
        answer_close_match = re.search(r"</answer>", response, re.IGNORECASE)
        start_index = sep_match.end()
        # Handle missing </answer> by going to end of string
        end_index = answer_close_match.start() if answer_close_match else len(response)
        potential_ranking_text = response[start_index:end_index].strip()

        # Extract the first valid-looking ranking from that part
        arrow_ranking = _hm_extract_arrow_ranking(potential_ranking_text)

        # Check if extracted ranking is valid for the number of statements
        if arrow_ranking and _hm_check_arrow_format(arrow_ranking, num_statements):
            ranking_arr = _parse_arrow_ranking_to_array(arrow_ranking, num_statements)
            if ranking_arr is None:
                # Parsing failed unexpectedly after validation passed
                # Use full response in error message
                explanation = f"INTERNAL_PARSING_ERROR: {response}"
            # else: explanation remains the original response
        else:
            # Valid format, but ranking extraction failed OR extracted ranking format invalid
            ranking_arr = None
            # Use full response in error message
            explanation = f"INCORRECT_ARROW_RANKING: {response}"

    else:
        # 2. Incorrect standard format - check for fallback 'final ranking:'
        final_match = re.search(r"final ranking:", response, re.IGNORECASE)
        if final_match:
            # Extract text after 'final ranking:' (up to newline or end)
            start_index = final_match.end()
            newline_match = re.search(r"\n", response[start_index:])
            end_index = (
                start_index + newline_match.start() if newline_match else len(response)
            )
            potential_ranking_text = response[start_index:end_index].strip()

            # Extract the first valid-looking ranking from that part
            arrow_ranking = _hm_extract_arrow_ranking(potential_ranking_text)

            # Check if extracted ranking is valid for the number of statements
            if arrow_ranking and _hm_check_arrow_format(arrow_ranking, num_statements):
                ranking_arr = _parse_arrow_ranking_to_array(
                    arrow_ranking, num_statements
                )
                if ranking_arr is None:
                    # Use full response in error message
                    explanation = f"INTERNAL_PARSING_ERROR: {response}"
                # else: explanation remains the original response
            else:
                # Fallback failed: 'final ranking:' present but no valid ranking found/validated after it
                ranking_arr = None
                # Original logic falls through to INCORRECT_TEMPLATE here
                # Use full response in error message
                explanation = f"INCORRECT_TEMPLATE: {response}"
        else:
            # 3. Incorrect template AND no 'final ranking:' found
            ranking_arr = None
            # Use full response in error message
            explanation = f"INCORRECT_TEMPLATE: {response}"

    # Final safety check
    if ranking_arr is None and not explanation.startswith(("INCORRECT_", "INTERNAL_")):
        # Use full response in error message
        explanation = f"UNKNOWN_PROCESSING_FAILURE: {response}"

    return ranking_arr, explanation


def _hm_predict_ranking_for_agent(
    model_identifier: str,
    question: str,
    opinion: str,
    statements: list[str],
    seed: int | None,
    num_retries_on_error: int = 1,
) -> tuple[np.ndarray | None, str]:
    """Generates prompt, calls LLM, and processes response to get agent ranking."""
    if not statements:
        return None, "NO_STATEMENTS_PROVIDED"

    prompt = _hm_generate_opinion_only_ranking_prompt(question, opinion, statements)

    current_seed = seed
    last_error = "Unknown error during ranking prediction."
    last_response = ""  # Store last response for debugging

    for attempt in range(num_retries_on_error + 1):
        print(
            f"      Attempt {attempt+1}/{num_retries_on_error+1} with seed {current_seed}..."
        )
        response = generate_text(
            model=model_identifier,
            system_prompt="",
            user_prompt=prompt,
            seed=current_seed,
            temperature=0.0,
        )
        last_response = response  # Store for potential error message

        if not response:
            last_error = "LLM_EMPTY_RESPONSE"
            print(f"      Attempt {attempt+1} failed: Empty response from LLM.")
            if current_seed is not None:
                current_seed += 1
            continue

        # Do NOT automatically append </answer> - let format check handle it

        ranking, explanation_or_error = _hm_process_ranking_response(
            response, len(statements)
        )

        if ranking is not None:
            print(f"      Attempt {attempt+1} successful.")
            return (
                ranking,
                explanation_or_error,  # Return explanation part on success
            )
        else:
            last_error = explanation_or_error  # Store the specific error message
            print(f"      Attempt {attempt+1} failed: {last_error}")
            if current_seed is not None:
                current_seed += 1

    print(f"      All {num_retries_on_error + 1} attempts failed.")
    # Include last response in final error for better debugging
    return (
        None,
        f"FAILED_AFTER_RETRIES: {last_error} (Last Response: {last_response[:100]}...)",
    )


# ==============================================================================
# Schulze Method Implementation (Adapted from provided code)
# ==============================================================================

# --- Helper functions for Schulze tie-breaking (adapted from utils) ---


def _hm_normalize_ranking(ranking: np.ndarray) -> np.ndarray:
    """Normalizes ranking so e.g. [0, 2, 5, 5] -> [0, 1, 2, 2]."""
    if ranking.ndim != 1:
        raise ValueError("The input array should be a single ranking so `ndim=1`")
    _, normalized_ranking = np.unique(ranking, return_inverse=True)
    return normalized_ranking


def _hm_is_untied_ranking(ranking: np.ndarray) -> bool:
    """Checks if the ranking is untied."""
    if ranking.ndim != 1:
        raise ValueError("The input array should be a single ranking so `ndim=1`")
    return np.unique(ranking).size == ranking.size


def _hm_untie_ranking_with_ballot(
    ranking: np.ndarray, ballot: np.ndarray
) -> np.ndarray:
    """Unties ranking with extra ballot and renormalizes rankings."""
    if ranking.ndim != 1:
        raise ValueError("The input array should be a single ranking so `ndim=1`")
    if ranking.shape != ballot.shape:
        raise ValueError("The ranking and ballot should have the same shape.")
    # We multiply the rankings with the number of candidates to ensure that we do
    # not change the order of the already sorted candidates. We then add a ballot
    # to untie the social ranking.
    # Ensure normalization happens before multiplication/addition
    normalized_ranking = _hm_normalize_ranking(ranking)
    normalized_ballot = _hm_normalize_ranking(ballot)
    untied_ranking = normalized_ranking * len(ranking) + normalized_ballot

    # Renormalize the social ranking to make sure ranks are consecutive.
    return _hm_normalize_ranking(untied_ranking)


# --- End Helper functions ---


def _schulze_check_rankings(rankings: np.ndarray):
    """Basic checks for the rankings array format."""
    if rankings.ndim != 2:
        raise ValueError(
            f"Rankings should be 2D [num_citizens, num_candidates], got shape {rankings.shape}"
        )
    if not np.issubdtype(rankings.dtype, np.integer):
        raise ValueError(f"Rankings should be integers, got {rankings.dtype}")
    # Check if ranks are within the expected range [0, num_candidates-1]
    num_candidates = rankings.shape[1]
    if np.any(rankings < 0) or np.any(rankings >= num_candidates):
        # Find the problematic value for a better error message
        invalid_rank = rankings[(rankings < 0) | (rankings >= num_candidates)][0]
        raise ValueError(
            f"Ranks must be between 0 and {num_candidates-1}. Found invalid rank: {invalid_rank}"
        )


def _schulze_compute_pairwise_defeats(rankings: np.ndarray) -> np.ndarray:
    """Computes the number of votes who prefer one over the other candidate.

    Args:
      rankings: Array of batched rankings with dimensions: [num_citizens,
        num_candidates]. Lower rank is better (0 is best).

    Returns:
      An array d[i, j] with the number of voters who prefer candidate i to candidate j.
        Dimensions [num_candidates, num_candidates].
    """
    num_citizens, num_candidates = rankings.shape
    pairwise_defeats = np.zeros((num_candidates, num_candidates), dtype=np.int32)
    for citizen_id in range(num_citizens):
        for idx in range(num_candidates):
            for idy in range(num_candidates):
                if idx == idy:
                    continue
                # Lower rank is better
                if rankings[citizen_id, idx] < rankings[citizen_id, idy]:
                    pairwise_defeats[idx, idy] += 1
    return pairwise_defeats


def _schulze_compute_strongest_path_strengths(
    pairwise_defeats: np.ndarray,
) -> np.ndarray:
    """Computes the strength of the strongest path between candidates.

    Args:
      pairwise_defeats: An array d[i, j] with the number of voters who prefer
          candidate i to candidate j. Dimensions [num_candidates, num_candidates].
    Returns:
      An array p[i, j] with the strength of the strongest path from candidate i to
      candidate j. Dimensions [num_candidates, num_candidates].
    """
    if (
        pairwise_defeats.ndim != 2
        or pairwise_defeats.shape[0] != pairwise_defeats.shape[1]
    ):
        raise ValueError(
            f"pairwise_defeats should be a square array, got shape {pairwise_defeats.shape}"
        )
    if np.any(np.diag(pairwise_defeats) != 0):
        # This shouldn't happen with the logic in _schulze_compute_pairwise_defeats
        # but check defensively.
        raise ValueError("pairwise_defeats should have an all zero diagonal.")

    num_candidates = pairwise_defeats.shape[0]
    # Initialize path_strengths p[i, j] for all i, j
    # p[i, j] = d[i, j] if d[i, j] > d[j, i]
    # p[i, j] = 0       if d[i, j] <= d[j, i]
    path_strengths = np.where(
        pairwise_defeats > pairwise_defeats.T, pairwise_defeats, 0
    )
    np.fill_diagonal(path_strengths, 0)  # Ensure diagonal is zero explicitly

    # Floyd-Warshall-like algorithm to find strongest paths
    for i in range(num_candidates):
        for j in range(num_candidates):
            if i == j:
                continue
            for k in range(num_candidates):
                if i == k or j == k:
                    continue
                # Strength of path j -> k is max(current strength, strength via intermediate i)
                # Strength via i is min(strength(j -> i), strength(i -> k))
                path_strengths[j, k] = max(
                    path_strengths[j, k],
                    min(path_strengths[j, i], path_strengths[i, k]),
                )

    return path_strengths


def _schulze_rank_candidates(path_strengths: np.ndarray) -> np.ndarray:
    """Rank the candidates based on strongest path strengths.

    A candidate C is better than D if p[C, D] > p[D, C].
    A candidate C is at least as good as D if p[C, D] >= p[D, C].
    Rank is determined by the number of other candidates they are at least as good as.
    Lower rank value means better overall rank (0 is best).

    Args:
      path_strengths: An array p[i, j] with the strength of the strongest path
        from candidate i to candidate j. Dimensions [num_candidates, num_candidates].
    Returns:
      An array with the aggregated social rank for each candidate (lower is better).
        Dimensions [num_candidates,]. Ties are possible.
    """
    if path_strengths.ndim != 2 or path_strengths.shape[0] != path_strengths.shape[1]:
        raise ValueError(
            f"path_strengths should be a square array, got shape {path_strengths.shape}"
        )
    if np.any(np.diag(path_strengths) != 0):
        # This shouldn't happen with the logic in _schulze_compute_strongest_path_strengths
        # but check defensively.
        raise ValueError("path_strengths should have an all zero diagonal.")

    num_candidates = path_strengths.shape[0]
    # Determine pairwise "at least as good as" relationship
    # pairwise_dominance[i, j] is True if i is at least as good as j (p[i, j] >= p[j, i])
    pairwise_dominance = path_strengths >= path_strengths.T

    # Count how many other candidates each candidate dominates (or is tied with)
    # Higher count means better rank.
    dominance_count = pairwise_dominance.sum(axis=1)

    # Rank based on dominance count. Higher count gets lower rank value (better rank).
    # We use unique with inverse=True to assign ranks, negating the count so higher counts map to lower ranks.
    # Example: counts [10, 12, 10] -> unique [-10, -12] -> ranks [0, 1, 0]
    _, social_ranking = np.unique(-dominance_count, return_inverse=True, axis=0)
    return social_ranking


def _schulze_aggregate_with_ties(
    rankings: np.ndarray,
) -> np.ndarray:
    """Aggregates rankings into a single social ranking with potential ties using Schulze."""
    # Check rankings format (allow ties by default in the check)
    # Note: The original check_rankings had more complex logic for allowed steps.
    # We'll rely on the basic checks here and the normalization within Schulze.
    # If stricter validation is needed later, _schulze_check_rankings can be enhanced.
    _schulze_check_rankings(rankings)  # Basic dimension/type/range checks

    pairwise_defeats = _schulze_compute_pairwise_defeats(rankings)
    strongest_path_strengths = _schulze_compute_strongest_path_strengths(
        pairwise_defeats
    )
    social_ranking = _schulze_rank_candidates(strongest_path_strengths)
    return social_ranking


def _aggregate_schulze(
    agent_rankings: dict[str, np.ndarray | None],
    num_candidates: int,
    seed: int | None = None,
    tie_breaking_method: str = "random",  # Add tie-breaking method
) -> np.ndarray | None:
    """
    Prepares rankings, calls the Schulze method, and handles tie-breaking.

    Args:
        agent_rankings: Dictionary mapping agent ID to their ranking array (lower is better)
                        or None if ranking failed.
        num_candidates: The total number of candidate statements.
        seed: Random seed for deterministic tie-breaking.
        tie_breaking_method: Method for breaking ties ('random' or 'ties_allowed').

    Returns:
        A numpy array representing the final social ranking (lower is better).
        If 'random' tie-breaking is used, the ranking should be untied.
        Returns None if aggregation is not possible.
    """
    valid_rankings_list = [r for r in agent_rankings.values() if r is not None]

    if not valid_rankings_list:
        print("  Schulze Aggregation Warning: No valid agent rankings provided.")
        return None

    # Stack valid rankings into a numpy array
    try:
        rankings_arr = np.stack(valid_rankings_list, axis=0)
        # Double check shape consistency
        if rankings_arr.shape[1] != num_candidates:
            print(
                f"  Schulze Aggregation Error: Ranking shape mismatch. Expected {num_candidates} candidates, found {rankings_arr.shape[1]}."
            )
            return None
    except ValueError as e:
        # Handle cases where rankings might have inconsistent shapes if not caught earlier
        print(f"  Schulze Aggregation Error: Could not stack rankings. {e}")
        print(
            f"  Individual ranking shapes: {[r.shape for r in valid_rankings_list if r is not None]}"
        )
        return None

    print(f"  Aggregating {rankings_arr.shape[0]} valid rankings using Schulze...")

    try:
        # Get the potentially tied social ranking
        social_ranking_tied = _schulze_aggregate_with_ties(rankings_arr)

        # Handle tie-breaking
        if tie_breaking_method == "ties_allowed" or _hm_is_untied_ranking(
            social_ranking_tied
        ):
            print("  No ties detected or ties allowed. Returning Schulze ranking.")
            return social_ranking_tied
        elif tie_breaking_method == "random":
            print("  Ties detected. Applying random tie-breaking...")
            # Use numpy's random generator for seed consistency
            rng = np.random.default_rng(seed)
            # Create a random ballot
            random_ballot = rng.permutation(num_candidates).astype(np.int32)
            # Untie the ranking using the random ballot
            social_ranking_untied = _hm_untie_ranking_with_ballot(
                social_ranking_tied, random_ballot
            )
            print(f"  Original tied ranking: {social_ranking_tied}")
            print(f"  Random ballot used: {random_ballot}")
            print(f"  Final untied ranking: {social_ranking_untied}")
            return social_ranking_untied
        # Add elif for 'tbrc' here if needed in the future
        else:
            print(
                f"  Warning: Unsupported tie_breaking_method '{tie_breaking_method}'. Returning tied ranking."
            )
            return social_ranking_tied

    except ValueError as e:
        print(f"  Schulze Aggregation Error: {e}")
        return None


def _generate_critique_prompt(issue: str, opinion: str, proposed_statement: str) -> str:
    """Generates a prompt for the LLM to critique a statement from an agent's perspective."""
    prompt = f"""
Task: You are acting as a participant in a deliberation process. Your goal is to critique a proposed consensus statement based *only* on your previously stated opinion. Evaluate how well the proposed statement reflects your views, pointing out specific agreements or disagreements.

Please think through this task step-by-step:

1.  Carefully re-read your original opinion to refresh your key points and priorities regarding the issue.
2.  Analyze the proposed consensus statement.
3.  Compare the proposed statement against your opinion. Does it capture your main points? Does it contradict anything you said? Does it omit something crucial from your perspective?
4.  Formulate a concise critique from your perspective. Focus on specific aspects of the proposed statement and how they relate to your opinion. If the statement is acceptable, explain why. If not, explain the specific shortcomings.

Provide your answer in the following format:
<answer>
[Your step-by-step reasoning comparing the statement to your opinion]
<sep>
[Your final critique of the proposed statement from your perspective]
</answer>

Example:
<answer>
1. My opinion emphasized the need for stricter regulations on industrial emissions as the primary way to improve air quality.
2. The proposed statement focuses on promoting public transport and green spaces.
3. While promoting public transport is good, the statement completely ignores my main point about industrial regulations. It feels incomplete and doesn't address the core issue I raised.
4. The critique should highlight this omission.
<sep>
While I agree that improving public transport is beneficial, this statement fails to address the critical issue of industrial emissions, which was the central point of my opinion. Without including measures to regulate industrial pollution, I cannot fully support this statement as a consensus.
</answer>

It is important to follow the template EXACTLY. So ALWAYS start with <answer>, then the explanation, then <sep> then only the final critique and then </answer>.

Below is the original question, your opinion, and the proposed consensus statement.

Question: {issue}

Your Opinion: {opinion}

Proposed Consensus Statement: {proposed_statement}

Provide your critique based *only* on your opinion:
<answer>
"""
    return prompt.strip()


def _generate_critique_for_agent(
    model_identifier: str,
    issue: str,
    opinion: str,
    winning_statement: str,
    seed: int | None,
) -> str | None:
    """Generates a critique for a winning statement from a specific agent's perspective."""

    prompt = _generate_critique_prompt(issue, opinion, winning_statement)

    response = generate_text(
        model=model_identifier,
        system_prompt="",
        user_prompt=prompt,
        seed=seed,
        temperature=1.0,
    )

    if not response:
        print("      Critique generation failed: Empty response from LLM.")
        return None

    # Use the existing processor to extract text after <sep>
    critique = _process_llm_response(response)

    if critique:
        return critique
    else:
        # _process_llm_response already prints warnings on failure
        print(
            f"      Critique generation failed: Could not parse response. Response snippet: {response[:100]}..."
        )
        return None


def _generate_revised_statement_prompt(
    issue: str,
    agent_opinions: dict,
    winning_statement: str,
    agent_critiques: dict,
) -> str:
    """Generates a prompt for the LLM to create a revised statement based on critiques."""

    opinions_list = list(agent_opinions.values())
    critiques_list = list(agent_critiques.values())

    prompt = f"""You are assisting a citizens' jury in forming a consensus opinion on an important question. The jury members have provided their individual opinions, a first draft of a consensus statement was created, and critiques of that draft were gathered. Your role is to generate a revised consensus statement that incorporates the feedback and aims to better represent the collective view of the jury. Ensure the revised statement does not conflict with the individual opinions.

Please think through this task step-by-step:

1. Carefully analyze the individual opinions, noting key themes, points of agreement, and areas of disagreement.
2. Review the previous draft consensus statement and identify its strengths and weaknesses.
3. Analyze the critiques of the previous draft, paying attention to specific suggestions and concerns raised by the jury members.
4. Based on the opinions, the previous draft, and the critiques, synthesize a revised consensus statement that addresses the concerns raised and better reflects the collective view of the jury. Ensure the statement is clear, concise, addresses the core issue posed in the question, and *does not conflict* with any of the individual opinions. Refer to specific opinion and critique numbers when making your revisions.
5. Keep the statement to less than 50 tokens.

Provide your answer in the following format:
<answer>
[Your step-by-step reasoning and explanation for the revised statement]
<sep>
[Revised consensus statement]
</answer>

Example:
<answer>
1. Opinions generally agree on the need for more green spaces (Opinions 1, 2, 3), but disagree on the specific location (Opinions 2 and 3 prefer the riverfront) and funding (Opinion 1 suggests a tax levy, Opinion 3 suggests private donations).
2. The previous draft suggested converting the old factory site into a park, but didn't address funding, which was a key concern in Critique 1.
3. Critiques highlighted the lack of funding details (Critique 1) and some preferred a different location (Critique 2 suggested the riverfront, echoing Opinion 2).
4. The revised statement proposes converting the old factory site into a park, funded by a combination of city funds and private donations (addressing Opinion 3 and Critique 1), and includes a plan for community input on park design and amenities. The factory site is chosen as a compromise location, as it avoids the higher costs associated with the riverfront development suggested in Opinion 2 and Critique 2.
<sep>
We propose converting the old factory site into a park, funded by a combination of city funds and private donations. We will actively seek community input on the park's design and amenities to ensure it meets the needs of our residents.
</answer>


Below you will find the question, the individual opinions, the previous draft consensus statement, and the critiques provided by the jury members.


Question: {issue}

Individual Opinions:
"""
    for i, opinion in enumerate(opinions_list):
        prompt += f"Opinion Person {i+1}: {opinion}\n"

    prompt += f"""
Previous Draft Consensus Statement: {winning_statement}

Critiques of the Previous Draft:
"""

    for i, critique in enumerate(critiques_list):
        prompt += f"Critique Person {i+1}: {critique}\n"

    return prompt.strip()


def _generate_multiple_revised_statements(
    model_identifier: str,
    issue: str,
    agent_opinions: dict,
    winning_statement: str,
    agent_critiques: dict,
    num_candidates: int,
    seed: int | None,
    num_retries: int = 1,
) -> list[str]:
    """Generates multiple revised statements for ranking."""

    prompt = _generate_revised_statement_prompt(
        issue=issue,
        agent_opinions=agent_opinions,
        winning_statement=winning_statement,
        agent_critiques=agent_critiques,
    )

    revised_statements = []
    current_seed = seed

    print(
        f"  Generating {num_candidates} revised statements using seed {current_seed}..."
    )

    for i in range(num_candidates):
        print(f"    Generating revised statement {i+1}/{num_candidates}...")
        statement_generated = (
            False  # Flag to track if statement was generated in this loop
        )

        # Try multiple times if needed
        for attempt in range(num_retries + 1):
            # Adjust seed calculation to ensure different seeds per attempt *within* the same candidate index 'i'
            if current_seed is not None:
                # Ensure unique seed for each attempt of each candidate
                used_seed = current_seed + i * (num_retries + 1) + attempt
            else:
                used_seed = None

            print(
                f"      Attempt {attempt+1}/{num_retries+1} with seed {used_seed}..."
            )  # Added print for seed used

            response = generate_text(
                model=model_identifier,
                system_prompt="",
                user_prompt=prompt,
                seed=used_seed,
            )

            statement = _process_llm_response(response)

            if statement:
                revised_statements.append(statement)
                print(
                    f"      Successfully generated revised statement {i+1} on attempt {attempt+1}."
                )
                statement_generated = True  # Set flag
                break  # Exit retry loop for this candidate
            else:
                # Warning printed by _process_llm_response
                print(
                    f"      Failed to extract revised statement {i+1} on attempt {attempt+1}."
                )
                if attempt < num_retries:
                    print(f"      Retrying...")
                # Seed will increment naturally if not None for the next attempt

        # Check if all attempts failed for this specific candidate 'i'
        if not statement_generated:
            print(
                f"      All attempts failed for revised statement {i+1}. Using previous winning statement as placeholder."
            )
            # Use the winning statement as a fallback to ensure we have enough candidates
            # Make sure 'winning_statement' is accessible here (it is passed as argument)
            revised_statements.append(winning_statement)

    # Increment base seed for the *next* candidate generation (outside the retry loop)
    # This was missing, leading to potential seed reuse if retries happened.
    # However, the seed calculation `used_seed = current_seed + i * (num_retries + 1) + attempt`
    # already ensures diversity across candidates 'i' and attempts.
    # So, explicitly incrementing current_seed here is not strictly necessary *if*
    # current_seed is only used as the base for this calculation within the function.
    # Let's keep the calculation as is, it handles diversity correctly.

    # Check if fewer candidates were generated than requested (including placeholders)
    if len(revised_statements) < num_candidates:
        print(
            f"Warning: Generated fewer statements ({len(revised_statements)}) than requested ({num_candidates}), even with placeholders."
        )

    print(f"  Finished generating {len(revised_statements)} revised statements.")
    return revised_statements
