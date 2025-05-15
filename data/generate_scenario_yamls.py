#!/usr/bin/env python3
"""
Script to generate YAML files for each scenario in the centroid issues.
"""

import os
import pandas as pd
import yaml

def load_centroid_issues(file_path="centroid_issues.csv"):
    """
    Load the centroid issues from the CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: The loaded data
    """
    return pd.read_csv(file_path)

def create_scenario_yaml(scenario_id, issue, opinions, output_dir):
    """
    Create a YAML file for a scenario.
    
    Args:
        scenario_id (int): The scenario ID
        issue (str): The issue text
        opinions (list): List of opinions
        output_dir (str): Directory to save the YAML file
    """
    # Filter out empty opinions
    opinions = [op for op in opinions if op]
    
    # Create agent opinions dictionary - include all opinions
    agent_opinions = {f"Agent {i+1}": opinion for i, opinion in enumerate(opinions)}
    
    # Create the config dictionary
    config = {
        "experiment_name": f"beam_search_scenario_{scenario_id}",
        "seed": 42,
        "num_seeds": 3,  # Run with multiple seeds for statistical significance
        
        "models": {
            "generation_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "evaluation_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        },
        
        "scenario": {
            "issue": issue,
            "agent_opinions": agent_opinions
        },
        
        "methods_to_run": ["beam_search"],
        
        "beam_search": {
            "beam_width": [1, 2, 5, 7],  # This is already a list in Python but YAML might render it differently
            "max_tokens": 200,
            "beta": 1.0,
            "log_level": "INFO",
            "api_delay": 0.5,
            "max_sampling_attempts": 10,
            "brushup": True,
            "brushup_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        },
        
        "output_dir": "results/"
    }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the YAML file
    yaml_path = os.path.join(output_dir, f"scenario_{scenario_id}.yaml")
    
    # Custom representer to handle lists differently based on content
    def represent_list(dumper, data):
        # Use flow style for beam_width, but block style for methods_to_run
        # If the list contains only integers, use flow style (for beam_width)
        if all(isinstance(item, int) for item in data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        # Otherwise, use block style (for methods_to_run)
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
    
    # Register the representer for lists
    yaml.add_representer(list, represent_list)
    
    with open(yaml_path, 'w') as f:
        # Add comment at the top of the file
        f.write("# Comprehensive Config for Comparing All Deliberation Methods\n")
        # Dump the YAML without the comment
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created YAML file for scenario {scenario_id} at {yaml_path}")

def main():
    """
    Main function to generate scenario YAML files.
    """
    # Define the output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "configs", "paper", "phase_2", "beam_search")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the centroid issues
    centroid_df = load_centroid_issues()
    
    # Get column names for opinions
    opinion_cols = [col for col in centroid_df.columns if col.startswith('opinion_')]
    
    # For each centroid issue
    for index, row in centroid_df.iterrows():
        scenario_id = int(row['cluster'])
        issue = row['issue']
        
        # Get all non-empty opinions for this issue
        opinions = [row[col] for col in opinion_cols if pd.notna(row[col]) and row[col]]
        
        # Create the YAML file
        create_scenario_yaml(scenario_id, issue, opinions, output_dir)
    
    print(f"Generated {len(centroid_df)} scenario YAML files in {output_dir}")

if __name__ == "__main__":
    main()