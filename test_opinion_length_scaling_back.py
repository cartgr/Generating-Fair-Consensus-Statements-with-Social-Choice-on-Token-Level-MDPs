#!/usr/bin/env python3
"""
Test how correlation scales with opinion length - TRUNCATING FROM THE BACK.
Does the END of the opinion matter more than the beginning?
"""

import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import time

sys.path.append("/Users/carterblair/0_Harvard/Research/consensus_statement_followup")
from get_likelihood import calculate_statement_likelihood


def truncate_opinion_from_back(opinion, fraction):
    """
    Keep only the LAST fraction of the opinion.
    E.g., fraction=0.25 keeps the last 25% of words.
    """
    words = opinion.split()
    n_words = max(1, int(len(words) * fraction))
    return " ".join(words[-n_words:])


def test_user_with_opinion_fraction(
    user_id, user_data, all_statements, issue, fraction, verbose=False
):
    """
    Test likelihood prediction using only the last fraction of the user's opinion.
    """

    opinion_full = user_data.iloc[0]["opinion"]
    opinion = truncate_opinion_from_back(opinion_full, fraction)

    if verbose:
        print(f"\n{'='*60}")
        print(f"User: {user_id} | Opinion fraction: LAST {fraction:.0%}")
        print(f"Full opinion length: {len(opinion_full.split())} words")
        print(f"Using last: {len(opinion.split())} words")
        print(f"Opinion excerpt: ...{opinion[:100]}")

    # Get user's ratings for all 5 statements
    ratings = []
    for statement in all_statements:
        stmt_data = user_data[user_data["statement"] == statement]
        if len(stmt_data) > 0:
            ratings.append(stmt_data.iloc[0]["rating"])
        else:
            ratings.append(3)  # Default middle rating if missing

    # Calculate likelihood for each statement
    likelihoods = []
    for i, statement in enumerate(all_statements):
        try:
            likelihood = calculate_statement_likelihood(
                issue=issue, agent_opinion=opinion, statement=statement
            )
            likelihoods.append(likelihood if likelihood else -10.0)
        except:
            likelihoods.append(-10.0)

    # Calculate correlation
    spearman_corr, _ = spearmanr(ratings, likelihoods)

    if verbose:
        print(f"Ratings: {ratings}")
        print(f"Likelihoods: {[f'{l:.3f}' for l in likelihoods]}")
        print(f"Spearman correlation: {spearman_corr:.3f}")

    return {
        "user_id": user_id,
        "fraction": fraction,
        "n_words": len(opinion.split()),
        "correlation": spearman_corr,
        "ratings": ratings,
        "likelihoods": [float(f"{l:.3f}") for l in likelihoods],
    }


def main():
    print("=" * 80)
    print("OPINION LENGTH SCALING - TRUNCATING FROM BACK")
    print("=" * 80)
    print("\nQuestion: Does the END of the opinion matter more than the beginning?")
    print("Testing with LAST 100%, 75%, 50%, and 25% of each user's opinion.\n")

    # Load data
    df = pd.read_csv("gsc_structured_for_likelihood.csv")
    all_statements = list(df["statement"].unique())

    print(f"Found {len(all_statements)} statements")

    # Test on users who rated all statements
    user_counts = df.groupby("user_id").size()
    complete_users = user_counts[user_counts == len(all_statements)].index
    test_users = list(complete_users)  # Test ALL users

    print(
        f"Testing on {len(test_users)} users (total API calls: {len(test_users) * 4 * 5})"
    )

    # Test with different opinion fractions (from back)
    fractions = [1.0, 0.75, 0.5, 0.25]
    issue = "What are your views on abortion?"

    results = []

    for fraction in fractions:
        print(f"\n{'='*80}")
        print(f"TESTING WITH LAST {fraction:.0%} OF OPINION TEXT")
        print(f"{'='*80}")

        for user_id in test_users:
            user_data = df[df["user_id"] == user_id]
            result = test_user_with_opinion_fraction(
                user_id, user_data, all_statements, issue, fraction, verbose=True
            )
            results.append(result)

    # Summary by fraction
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")

    results_df = pd.DataFrame(results)

    print("\nCorrelation by Opinion Length (from back):")
    print(f"{'Opinion %':<12} {'Mean Corr':<12} {'Std':<12} {'Mean Words':<12}")
    print("-" * 48)

    for fraction in fractions:
        fraction_results = results_df[results_df["fraction"] == fraction]
        mean_corr = fraction_results["correlation"].mean()
        std_corr = fraction_results["correlation"].std()
        mean_words = fraction_results["n_words"].mean()

        print(
            f"Last {fraction:>4.0%}   {mean_corr:>8.3f}     {std_corr:>8.3f}     {mean_words:>8.1f}"
        )

    # Statistical test: does correlation increase with opinion length?
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    corr_100 = results_df[results_df["fraction"] == 1.0]["correlation"].mean()
    corr_25 = results_df[results_df["fraction"] == 0.25]["correlation"].mean()

    improvement = corr_100 - corr_25

    if improvement > 0.1:
        print(
            f"\n✓ Strong effect: Full opinion improves correlation by {improvement:.3f}"
        )
        print("  → Earlier parts of the opinion add substantial information")
    elif improvement > 0.05:
        print(
            f"\n✓ Moderate effect: Full opinion improves correlation by {improvement:.3f}"
        )
        print("  → Some benefit from earlier parts of opinion")
    elif improvement > 0:
        print(
            f"\n~ Weak effect: Full opinion improves correlation by {improvement:.3f}"
        )
        print("  → Minimal benefit from earlier parts")
    else:
        print(
            f"\n✗ No improvement: Full opinion changes correlation by {improvement:.3f}"
        )
        print("  → Conclusions/summaries at the end are most informative!")

    # Save results
    results_df.to_csv("gsc_opinion_length_scaling_back_results.csv", index=False)
    print(f"\n✅ Results saved to: gsc_opinion_length_scaling_back_results.csv")


if __name__ == "__main__":
    main()
