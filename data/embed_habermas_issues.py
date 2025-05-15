#!/usr/bin/env python3
"""
Script to embed all unique issues from the Habermas dataset.
"""
import json
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add the project root to the path so we can import from src
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.utils import get_embedding
from data.load_habermas_data import get_unique_issues


# This function is no longer needed since the logic is now in main()
# Keeping it as a stub for backward compatibility
def embed_unique_issues(embedding_model="BAAI/bge-large-en-v1.5"):
    """
    Get and embed all unique issues from the Habermas dataset.
    
    This functionality has been moved to the main() function with 
    resumable processing and better error handling.
    
    Args:
        embedding_model (str): Model to use for generating embeddings
        
    Returns:
        list: List of dictionaries with issue text and embeddings
    """
    print("This function is deprecated. Using main() instead.")
    return main()


def save_progress(results, json_output, csv_output):
    """Save current progress to both JSON and CSV files."""
    # Save JSON with embeddings
    with open(json_output, 'w') as f:
        json.dump(results, f)
    print(f"Saved embeddings to {json_output}")
    
    # Save CSV with just the issues (no embeddings)
    df = pd.DataFrame([{"issue": r["issue"], "has_embedding": r["embedding"] is not None} 
                       for r in results])
    df.to_csv(csv_output, index=False)
    print(f"Saved issue text data to {csv_output}")


def main(embedding_model="BAAI/bge-large-en-v1.5"):
    """
    Main function to execute the embedding.
    
    Args:
        embedding_model (str): Model to use for generating embeddings
        
    Returns:
        list: List of dictionaries with issue text and embeddings
    """
    # Define output filenames
    json_output = "habermas_issue_embeddings.json"
    csv_output = "habermas_issues.csv"
    
    # Check if we already have some embeddings saved
    try:
        with open(json_output, 'r') as f:
            existing_results = json.load(f)
            print(f"Loaded {len(existing_results)} existing embeddings")
            
            # Get set of issues we've already embedded
            embedded_issues = {r["issue"] for r in existing_results}
            print(f"Found {len(embedded_issues)} issues already embedded")
            
            # Get all unique issues
            all_issues = set(get_unique_issues())
            print(f"Total unique issues: {len(all_issues)}")
            
            # Get issues we still need to embed
            remaining_issues = all_issues - embedded_issues
            print(f"Need to embed {len(remaining_issues)} more issues")
            
            if not remaining_issues:
                print("All issues already embedded!")
                return existing_results
    except (FileNotFoundError, json.JSONDecodeError):
        print("No existing embeddings found, starting from scratch")
        existing_results = []
        remaining_issues = None  # Will use get_unique_issues() directly
    
    # Start or continue embedding
    if remaining_issues is not None:
        # Continue from where we left off
        results = existing_results
        issues_to_embed = list(remaining_issues)
    else:
        # Start fresh
        results = []
        issues_to_embed = get_unique_issues()
    
    # Embed each issue
    try:
        for i, issue in enumerate(tqdm(issues_to_embed, desc="Embedding issues")):
            print(f"Embedding: {issue[:100]}...")
            
            # Get embedding
            embedding = get_embedding(issue, model=embedding_model)
            
            # Save result
            result = {
                "issue": issue,
                "embedding": embedding if embedding is not None else None
            }
            results.append(result)
            
            # Save progress periodically
            if (i + 1) % 50 == 0:
                print(f"Saving progress after {i + 1}/{len(issues_to_embed)} issues...")
                save_progress(results, json_output, csv_output)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current progress...")
    except Exception as e:
        print(f"\nError occurred: {e}. Saving current progress...")
    finally:
        # Save final results
        save_progress(results, json_output, csv_output)
    
    # Count issues with valid embeddings
    valid_embeddings = sum(1 for r in results if r["embedding"] is not None)
    print(f"Successfully embedded {valid_embeddings}/{len(results)} issues")
    
    return results


if __name__ == "__main__":
    main()