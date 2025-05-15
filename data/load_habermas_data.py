#!/usr/bin/env python3
"""
Utility script for loading and processing Habermas Machine datasets.
"""

import pandas as pd
import os
import random
import numpy as np


def load_round_survey_responses(simplified=True):
    """
    Load the round survey responses dataset.

    Args:
        simplified (bool): If True, returns only issue, opinion, and consensus statement columns.
                          If False, returns the full dataset.

    Returns:
        pandas.DataFrame: The loaded dataset
    """
    file_path = os.path.join(
        os.path.dirname(__file__), "data/hm_all_round_survey_responses.parquet"
    )
    df = pd.read_parquet(file_path)

    if simplified:
        # Return just the columns needed for (issue, opinion, consensus statement)
        return df[["question.text", "opinion.text", "candidate.text"]].rename(
            columns={
                "question.text": "issue",
                "opinion.text": "opinion",
                "candidate.text": "consensus_statement",
            }
        )
    return df


def load_candidate_comparisons():
    """Load the candidate comparisons dataset."""
    file_path = os.path.join(
        os.path.dirname(__file__), "hm_all_candidate_comparisons.parquet"
    )
    return pd.read_parquet(file_path)


def load_preference_rankings():
    """Load the preference rankings dataset."""
    file_path = os.path.join(
        os.path.dirname(__file__), "hm_all_final_preference_rankings.parquet"
    )
    return pd.read_parquet(file_path)


def load_position_statement_ratings():
    """Load the position statement ratings dataset."""
    file_path = os.path.join(
        os.path.dirname(__file__), "hm_all_position_statement_ratings.parquet"
    )
    return pd.read_parquet(file_path)


def get_unique_issues():
    """Get a list of all unique issues in the dataset."""
    df = load_round_survey_responses(simplified=True)
    return df["issue"].unique()


def get_data_for_issue(issue_text):
    """
    Get all opinions and consensus statements for a specific issue.

    Args:
        issue_text (str): The exact text of the issue to search for

    Returns:
        pandas.DataFrame: A dataframe with only rows matching the specified issue
    """
    df = load_round_survey_responses(simplified=True)
    return df[df["issue"] == issue_text]


def get_example_dataset(num_issues=3, opinions_per_issue=2):
    """
    Create a small sample dataset with a specified number of issues and opinions per issue.

    Args:
        num_issues (int): Number of unique issues to include
        opinions_per_issue (int): Number of opinions to include per issue

    Returns:
        pandas.DataFrame: A sample dataset
    """
    df = load_round_survey_responses(simplified=True)
    unique_issues = df["issue"].unique()[:num_issues]

    sample_data = []
    for issue in unique_issues:
        issue_df = df[df["issue"] == issue]
        opinions = issue_df["opinion"].unique()[:opinions_per_issue]
        consensus = issue_df["consensus_statement"].iloc[0]

        for opinion in opinions:
            if pd.notna(opinion) and opinion != "No opinion was provided.":
                sample_data.append(
                    {
                        "issue": issue,
                        "opinion": opinion,
                        "consensus_statement": consensus,
                    }
                )

    return pd.DataFrame(sample_data)


def save_random_issues_to_txt(num_issues=5, seed=42, output_file="sample_issues.txt"):
    """
    Sample random issues with all their opinions and consensus statements and save to a txt file.

    Args:
        num_issues (int): Number of random issues to sample
        seed (int): Random seed for reproducibility
        output_file (str): Path to the output text file
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    df = load_round_survey_responses(simplified=True)
    all_issues = get_unique_issues()

    # Sample random issues
    sampled_issues = random.sample(list(all_issues), num_issues)

    with open(output_file, "w") as f:
        for i, issue in enumerate(sampled_issues, 1):
            issue_data = get_data_for_issue(issue)

            # Get unique non-empty opinions
            opinions = [
                op
                for op in issue_data["opinion"].unique()
                if pd.notna(op) and op != "No opinion was provided."
            ]

            # Get the consensus statement (should be the same for all rows with this issue)
            consensus = issue_data["consensus_statement"].iloc[0]

            # Write to file
            f.write(f"EXAMPLE {i}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"ISSUE:\n{issue}\n\n")

            f.write(f"OPINIONS ({len(opinions)}):\n")
            for j, opinion in enumerate(opinions, 1):
                f.write(f"{j}. {opinion}\n\n")

            f.write(f"CONSENSUS STATEMENT:\n{consensus}\n\n\n")

    print(f"Saved {num_issues} random issues to {output_file}")


if __name__ == "__main__":
    # Print total number of unique issues
    all_issues = get_unique_issues()
    print(f"Total number of unique issues in the dataset: {len(all_issues)}")

    # Save 5 random issues with seed 42
    save_random_issues_to_txt(num_issues=5, seed=42)

    # Example usage
    print("\nLoading example dataset...")
    example_df = get_example_dataset()
    print(f"Sample dataset shape: {example_df.shape}")

    # Display first issue and related data
    first_issue = example_df["issue"].iloc[0]
    print(f"\nExample issue: {first_issue}")

    issue_data = get_data_for_issue(first_issue)
    print(f"Found {len(issue_data)} opinions for this issue")

    print("\nExample opinions:")
    for i, opinion in enumerate(issue_data["opinion"].iloc[:7]):
        if pd.notna(opinion) and opinion != "No opinion was provided.":
            print(f"\nOpinion {i+1}:")
            print(opinion[:200] + "..." if len(opinion) > 200 else opinion)

    print("\nExample consensus statement:")
    print(
        example_df["consensus_statement"].iloc[0][:200] + "..."
        if len(example_df["consensus_statement"].iloc[0]) > 200
        else example_df["consensus_statement"].iloc[0]
    )
