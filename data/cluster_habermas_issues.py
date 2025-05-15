#!/usr/bin/env python3
"""
Script to cluster Habermas issues based on their embeddings.
This version filters the dataset to only include issues that have 5 or fewer
total agent opinions. Issues with more than 5 opinions are excluded completely.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path if running from this directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_habermas_data import get_data_for_issue


def load_issue_embeddings(json_file="habermas_issue_embeddings.json"):
    """
    Load issue embeddings from JSON file.

    Args:
        json_file (str): Path to the JSON file with embeddings

    Returns:
        list: List of issue data with embeddings
    """
    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        valid_embeddings = sum(1 for item in data if item.get("embedding") is not None)
        print(
            f"Loaded embeddings for {len(data)} issues ({valid_embeddings} with valid embeddings)"
        )
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading embeddings: {e}")
        return []


def extract_embeddings(issue_data, max_opinions=5):
    """
    Extract embeddings from issue data, selecting ONLY issues with at most max_opinions.
    
    This function filters the dataset to only include issues where the total number
    of agent opinions is less than or equal to max_opinions. Issues with more 
    opinions are completely excluded from clustering.

    Args:
        issue_data (list): List of issue data with embeddings
        max_opinions (int): Maximum number of agent opinions per issue (default: 5)

    Returns:
        tuple: (numpy array of embeddings, list of issue info)
    """
    embeddings = []
    issue_info = []
    included_count = 0
    excluded_count = 0
    
    print(f"Filtering issues to those with {max_opinions} or fewer opinions...")
    
    # Create a progress bar
    for item in tqdm(issue_data, desc="Processing issues", unit="issue"):
        if item.get("embedding"):
            issue_text = item.get("issue", "")
            
            # Get all opinions for this issue to count them
            issue_df = get_data_for_issue(issue_text)
            
            # Filter out empty or "No opinion was provided" opinions
            opinions = [
                op
                for op in issue_df["opinion"].unique()
                if pd.notna(op) and op != "No opinion was provided."
            ]
            
            # Only include issues with max_opinions or fewer total opinions
            # (we don't truncate opinions, we just exclude the entire issue if it has too many)
            if len(opinions) <= max_opinions:
                # Convert embedding to numpy array if needed
                embedding = item["embedding"]
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                embeddings.append(embedding)

                # Store issue info with ALL opinions
                issue_info.append({"issue": issue_text, "num_opinions": len(opinions)})
                included_count += 1
            else:
                excluded_count += 1

    # After processing, show summary
    print(f"\nSummary of filtering results:")
    print(f"- Selected {len(embeddings)} issues with {max_opinions} or fewer opinions")
    print(f"- Excluded {excluded_count} issues with more than {max_opinions} opinions")
    
    # Display some examples of included issues
    if included_count > 0:
        print("\nSample of included issues:")
        sample_size = min(3, included_count)
        for i in range(sample_size):
            issue_text = issue_info[i]["issue"]
            num_opinions = issue_info[i]["num_opinions"]
            print(f"- Issue with {num_opinions} opinions: {issue_text[:50]}...")
    
    return np.array(embeddings), issue_info


def cluster_embeddings(embeddings, n_clusters=5, seed=42):
    """
    Cluster the embeddings using K-means.

    Args:
        embeddings (numpy.ndarray): Array of embeddings
        n_clusters (int): Number of clusters
        seed (int): Random seed

    Returns:
        tuple: (numpy.ndarray of cluster labels, numpy.ndarray of cluster centers)
    """
    # Standardize the embeddings
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_embeddings)

    return cluster_labels, kmeans.cluster_centers_


def find_cluster_centroids(embeddings, issue_info, cluster_labels, cluster_centers):
    """
    Find the issues closest to each cluster centroid.

    Args:
        embeddings (numpy.ndarray): Array of embeddings
        issue_info (list): List of issue info dictionaries
        cluster_labels (numpy.ndarray): Cluster assignments
        cluster_centers (numpy.ndarray): Cluster centers

    Returns:
        list: List of centroid issues for each cluster
    """
    # For each cluster center, find the closest issue
    closest_indices, _ = pairwise_distances_argmin_min(cluster_centers, embeddings)

    # Get the closest issue for each cluster
    centroid_issues = [issue_info[idx] for idx in closest_indices]

    return centroid_issues


def save_cluster_results(issue_info, cluster_labels, centroid_issues, output_file):
    """
    Save the clustering results to a CSV file.

    Args:
        issue_info (list): List of issue info dictionaries
        cluster_labels (numpy.ndarray): Cluster assignments
        centroid_issues (list): List of centroid issues
        output_file (str): Path to output file
    """
    # Create a DataFrame with issue info and cluster assignments
    df = pd.DataFrame(issue_info)
    df["cluster"] = cluster_labels

    # Mark centroid issues
    df["is_centroid"] = False
    centroid_texts = [issue["issue"] for issue in centroid_issues]
    df.loc[df["issue"].isin(centroid_texts), "is_centroid"] = True

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved clustering results to {output_file}")

    # Print cluster info
    print("\nCluster distribution:")
    print(df["cluster"].value_counts())

    # Print opinion count statistics
    if "num_opinions" in df.columns:
        print("\nOpinion count statistics:")
        print(df["num_opinions"].describe())

    print("\nCentroid issues:")
    for i, issue in enumerate(centroid_issues):
        num_opinions = issue.get("num_opinions", "unknown")
        print(f"Cluster {i}: ({num_opinions} opinions) {issue['issue'][:100]}...")


def save_centroid_issues_to_file(
    centroid_issues,
    output_file="centroid_issues.csv",
    txt_output_file="centroid_issues.txt",
):
    """
    Save the centroid issues to dedicated CSV and TXT files.

    Args:
        centroid_issues (list): List of centroid issues
        output_file (str): Path to output CSV file
        txt_output_file (str): Path to output TXT file with issues and opinions
    """
    # Create a DataFrame with just the centroid issues
    df = pd.DataFrame(
        [
            {"cluster": i, "issue": issue["issue"]}
            for i, issue in enumerate(centroid_issues)
        ]
    )

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved {len(centroid_issues)} centroid issues to {output_file}")

    # Save detailed information (issue + agent opinions) to text file
    with open(txt_output_file, "w") as f:
        for i, issue_data in enumerate(centroid_issues):
            issue = issue_data["issue"]
            f.write(f"CLUSTER {i+1}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"ISSUE:\n{issue}\n\n")

            # Get all opinions for this issue
            issue_df = get_data_for_issue(issue)

            # Filter out empty or "No opinion was provided" opinions
            opinions = [
                op
                for op in issue_df["opinion"].unique()
                if pd.notna(op) and op != "No opinion was provided."
            ]

            f.write(f"OPINIONS ({len(opinions)}):\n")
            for j, opinion in enumerate(opinions, 1):  # Show all available opinions
                f.write(f"{j}. {opinion}\n\n")

            # Get consensus statement
            if "consensus_statement" in issue_df.columns and len(issue_df) > 0:
                consensus = issue_df["consensus_statement"].iloc[0]
                f.write(f"CONSENSUS STATEMENT:\n{consensus}\n\n")

            f.write(f"\n\n")

    print(f"Saved detailed centroid issues with opinions to {txt_output_file}")


def main(n_clusters=5, seed=42, max_opinions=5):
    """
    Main function to execute the clustering pipeline.

    Args:
        n_clusters (int): Number of clusters
        seed (int): Random seed
        max_opinions (int): Maximum number of agent opinions to include
    """
    print(f"Starting Habermas issue clustering with seed {seed} and max {max_opinions} opinions")

    # Load the embeddings
    issue_data = load_issue_embeddings()
    if not issue_data:
        print("No embeddings found. Run embed_habermas_issues.py first.")
        return

    # Extract embeddings, filtering issues with more than max_opinions
    embeddings, issue_info = extract_embeddings(issue_data, max_opinions=max_opinions)

    if len(embeddings) < n_clusters:
        print(
            f"Not enough issue embeddings ({len(embeddings)}) for {n_clusters} clusters"
        )
        n_clusters = max(2, len(embeddings))
        print(f"Reducing to {n_clusters} clusters")
    
    if len(embeddings) == 0:
        print("No issues found with the specified number of opinions. Exiting.")
        return

    # Cluster issues
    print("\nPerforming K-means clustering...")
    with tqdm(total=4, desc="Clustering steps") as pbar:
        # Step 1: Standardize data
        pbar.set_description("Standardizing data")
        # Cluster issues
        cluster_labels, cluster_centers = cluster_embeddings(
            embeddings, n_clusters=n_clusters, seed=seed
        )
        pbar.update(1)
        
        # Step 2: Find centroids
        pbar.set_description("Finding cluster centroids")
        centroid_issues = find_cluster_centroids(
            embeddings, issue_info, cluster_labels, cluster_centers
        )
        pbar.update(1)
        
        # Step 3: Save cluster results
        pbar.set_description("Saving cluster results")
        save_cluster_results(
            issue_info, cluster_labels, centroid_issues, "habermas_issue_clusters.csv"
        )
        pbar.update(1)
        
        # Step 4: Save centroid issues
        pbar.set_description("Saving centroid issues")
        save_centroid_issues_to_file(
            centroid_issues,
            output_file="centroid_issues.csv",
            txt_output_file="centroid_issues.txt",
        )
        pbar.update(1)

    print("\nCompleted Habermas issue clustering successfully")


if __name__ == "__main__":
    # Set a different random seed for new clustering results
    SEED = 43
    MAX_OPINIONS = 5  # Only include issues with 5 or fewer agent opinions
    main(n_clusters=5, seed=SEED, max_opinions=MAX_OPINIONS)
