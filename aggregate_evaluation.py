#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_aggregate_directory(result_dir: Path) -> Path:
    """
    Creates an aggregate directory in the evaluation folder.
    
    Args:
        result_dir: Path to the result directory
        
    Returns:
        Path to the created aggregate directory
    """
    aggregate_dir = result_dir / "evaluation" / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created aggregate directory: {aggregate_dir}")
    return aggregate_dir

def get_seed_directories(model_dir: Path) -> List[Path]:
    """
    Returns all seed directories in a model directory.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        List of seed directory paths
    """
    return [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]

def aggregate_model_metrics(
    result_dir: Path, 
    model_name: str, 
    aggregate_dir: Path
) -> pd.DataFrame:
    """
    Aggregates metrics for a specific evaluation model across all seeds.
    
    Args:
        result_dir: Path to the result directory
        model_name: Name of the model to aggregate
        aggregate_dir: Path to the aggregate directory
        
    Returns:
        DataFrame with aggregated metrics
    """
    model_dir = result_dir / "evaluation" / model_name
    
    if not model_dir.exists():
        logger.warning(f"Model directory not found: {model_dir}")
        return pd.DataFrame()
    
    # Get all seed directories
    seed_dirs = get_seed_directories(model_dir)
    
    if not seed_dirs:
        logger.warning(f"No seed directories found in {model_dir}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(seed_dirs)} seed directories for {model_name}")
    
    # Collect all evaluation results
    all_results = []
    
    for seed_dir in seed_dirs:
        eval_file = seed_dir / "evaluation_results.csv"
        
        if not eval_file.exists():
            logger.warning(f"Evaluation results file not found: {eval_file}")
            continue
        
        try:
            df = pd.read_csv(eval_file)
            all_results.append(df)
            logger.info(f"Read {len(df)} rows from {eval_file}")
        except Exception as e:
            logger.error(f"Error reading {eval_file}: {e}")
    
    if not all_results:
        logger.warning(f"No evaluation results found for {model_name}")
        return pd.DataFrame()
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Create a unique identifier for each method based on method name and parameters
    combined_df['method_id'] = combined_df.apply(
        lambda row: create_method_identifier_from_row(row), axis=1
    )
    
    # List of metrics to aggregate
    perplexity_metrics = [
        'egalitarian_welfare_perplexity', 
        'utilitarian_welfare_perplexity'
    ]
    
    cosine_metrics = [
        'egalitarian_welfare_cosine', 
        'utilitarian_welfare_cosine'
    ]
    
    # Get agent-specific metrics
    agent_columns = [col for col in combined_df.columns if col.startswith('perplexity_Agent')]
    perplexity_metrics.extend(agent_columns)
    
    agent_columns = [col for col in combined_df.columns if col.startswith('cosine_similarity_Agent')]
    cosine_metrics.extend(agent_columns)
    
    # All metrics to aggregate
    all_metrics = perplexity_metrics + cosine_metrics
    
    # Group by method identifier and calculate mean and std
    aggregated_metrics = []
    
    for method_id, group_df in combined_df.groupby('method_id'):
        # Extract method name and parameters
        method_row = group_df.iloc[0]
        method_name = method_row['method']
        
        # Get all parameter columns
        param_cols = [col for col in group_df.columns if col.startswith('param_')]
        
        # Create base row with method and parameters
        agg_row = {
            'method': method_name,
            'method_id': method_id,
        }
        
        # Add parameters
        for param_col in param_cols:
            param_values = group_df[param_col].dropna().unique()
            if len(param_values) == 1:
                agg_row[param_col] = param_values[0]
        
        # Calculate mean and std for each metric
        for metric in all_metrics:
            if metric in group_df.columns:
                values = group_df[metric].dropna().values
                if len(values) > 0:
                    agg_row[f'{metric}_mean'] = np.mean(values)
                    agg_row[f'{metric}_std'] = np.std(values)
        
        aggregated_metrics.append(agg_row)
    
    # Create DataFrame from aggregated metrics
    agg_df = pd.DataFrame(aggregated_metrics)
    
    # Save model-specific aggregated results
    model_agg_dir = aggregate_dir / model_name
    model_agg_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = model_agg_dir / "aggregated_metrics.csv"
    agg_df.to_csv(output_file, index=False)
    logger.info(f"Saved aggregated metrics for {model_name} to {output_file}")
    
    return agg_df

def aggregate_llm_judge_metrics(
    result_dir: Path, 
    aggregate_dir: Path
) -> pd.DataFrame:
    """
    Aggregates LLM judge metrics across all seeds.
    
    Args:
        result_dir: Path to the result directory
        aggregate_dir: Path to the aggregate directory
        
    Returns:
        DataFrame with aggregated LLM judge metrics
    """
    llm_judge_dir = result_dir / "evaluation" / "llm_judge"
    
    if not llm_judge_dir.exists():
        logger.warning(f"LLM judge directory not found: {llm_judge_dir}")
        return pd.DataFrame()
    
    # Get all seed directories
    seed_dirs = get_seed_directories(llm_judge_dir)
    
    if not seed_dirs:
        logger.warning(f"No seed directories found in {llm_judge_dir}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(seed_dirs)} seed directories for LLM judge")
    
    # Collect all ranking results
    all_rankings = []
    
    for seed_dir in seed_dirs:
        ranking_file = seed_dir / "ranking_results.csv"
        
        if not ranking_file.exists():
            logger.warning(f"Ranking results file not found: {ranking_file}")
            continue
        
        try:
            df = pd.read_csv(ranking_file)
            all_rankings.append(df)
            logger.info(f"Read {len(df)} rows from {ranking_file}")
        except Exception as e:
            logger.error(f"Error reading {ranking_file}: {e}")
    
    if not all_rankings:
        logger.warning("No LLM judge ranking results found")
        return pd.DataFrame()
    
    # Combine all rankings
    combined_df = pd.concat(all_rankings, ignore_index=True)
    
    # Create a unique identifier for each method based on method name and parameters
    # Note: LLM judge results already have method_with_params column
    combined_df['method_id'] = combined_df.apply(
        lambda row: create_method_identifier_from_row(row), axis=1
    )
    
    # List of metrics to aggregate
    ranking_metrics = ['min_rank', 'max_rank', 'avg_rank']
    
    # Get agent-specific rank columns
    agent_rank_columns = [col for col in combined_df.columns if col.startswith('rank_Agent')]
    ranking_metrics.extend(agent_rank_columns)
    
    # Group by method identifier and calculate mean and std
    aggregated_metrics = []
    
    for method_id, group_df in combined_df.groupby('method_id'):
        # Extract method name and parameters
        method_row = group_df.iloc[0]
        method_name = method_row['method']
        
        # Get all parameter columns
        param_cols = [col for col in group_df.columns if col.startswith('param_')]
        
        # Create base row with method and parameters
        agg_row = {
            'method': method_name,
            'method_id': method_id,
        }
        
        # Add parameters
        for param_col in param_cols:
            param_values = group_df[param_col].dropna().unique()
            if len(param_values) == 1:
                agg_row[param_col] = param_values[0]
        
        # Calculate mean and std for each metric
        for metric in ranking_metrics:
            if metric in group_df.columns:
                values = group_df[metric].dropna().values
                if len(values) > 0:
                    agg_row[f'{metric}_mean'] = np.mean(values)
                    agg_row[f'{metric}_std'] = np.std(values)
        
        aggregated_metrics.append(agg_row)
    
    # Create DataFrame from aggregated metrics
    agg_df = pd.DataFrame(aggregated_metrics)
    
    # Save LLM judge aggregated results
    llm_judge_agg_dir = aggregate_dir / "llm_judge"
    llm_judge_agg_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = llm_judge_agg_dir / "aggregated_rankings.csv"
    agg_df.to_csv(output_file, index=False)
    logger.info(f"Saved aggregated LLM judge metrics to {output_file}")
    
    return agg_df

def create_method_identifier_from_row(row: pd.Series) -> str:
    """
    Creates a unique identifier for a method based on the method name and parameters.
    Uses the standardized create_method_identifier function from utils.

    Args:
        row: DataFrame row with method information

    Returns:
        A unique identifier string
    """
    method = row['method']

    # Get all parameter columns and values
    param_cols = [col for col in row.index if col.startswith('param_')]
    params_dict = {}
    for param in param_cols:
        if pd.notna(row[param]):
            params_dict[param] = row[param]

    # Import the standardized function
    import sys
    sys.path.append('.')  # Ensure we can import from the root directory

    from src.utils import create_method_identifier
    return create_method_identifier(
        method_name=method,
        params_dict=params_dict,
        include_seed=False  # We're aggregating across seeds
    )

def combine_metrics(
    eval_dfs: Dict[str, pd.DataFrame], 
    llm_judge_df: pd.DataFrame,
    aggregate_dir: Path
) -> pd.DataFrame:
    """
    Combines metrics from different evaluation models and LLM judge.
    
    Args:
        eval_dfs: Dictionary mapping model names to their aggregated metrics
        llm_judge_df: DataFrame with aggregated LLM judge metrics
        aggregate_dir: Path to the aggregate directory
        
    Returns:
        DataFrame with combined metrics
    """
    # If no evaluation metrics, return empty DataFrame
    if not eval_dfs and llm_judge_df.empty:
        logger.warning("No metrics to combine")
        return pd.DataFrame()
    
    # Start with the first evaluation model, or an empty DataFrame if none
    combined_df = next(iter(eval_dfs.values())) if eval_dfs else pd.DataFrame()
    
    # If we started with an empty DataFrame and have LLM judge metrics, use that
    if combined_df.empty and not llm_judge_df.empty:
        combined_df = llm_judge_df
    
    # If we have a combined DataFrame and LLM judge metrics, merge them
    elif not combined_df.empty and not llm_judge_df.empty:
        # Only select columns that don't duplicate method info
        llm_cols = [
            col for col in llm_judge_df.columns 
            if not col in ['method', 'param_n', 'param_num_candidates'] 
            and 'method_id' not in col
        ]
        combined_df = pd.merge(
            combined_df, 
            llm_judge_df[['method_id'] + llm_cols], 
            on='method_id', 
            how='outer'
        )
    
    # If we have multiple evaluation models, merge their metrics
    if len(eval_dfs) > 1:
        for model_name, df in list(eval_dfs.items())[1:]:
            # Only select columns that don't duplicate method info
            model_cols = [
                col for col in df.columns 
                if not col in ['method', 'param_n', 'param_num_candidates'] 
                and 'method_id' not in col
            ]
            # Add model name prefix to avoid column name conflicts
            model_df = df[['method_id'] + model_cols].copy()
            model_df.columns = [
                f"{model_name}_{col}" if col != 'method_id' else col 
                for col in model_df.columns
            ]
            combined_df = pd.merge(
                combined_df, 
                model_df, 
                on='method_id', 
                how='outer'
            )
    
    # Save combined metrics
    output_file = aggregate_dir / "combined_metrics.csv"
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Saved combined metrics to {output_file}")
    
    # Create a simplified version with just the key metrics
    simplified_df = create_simplified_metrics(combined_df, eval_dfs.keys())
    
    # Save simplified metrics
    output_file = aggregate_dir / "simplified_metrics.csv"
    simplified_df.to_csv(output_file, index=False)
    logger.info(f"Saved simplified metrics to {output_file}")
    
    return combined_df

def create_simplified_metrics(
    combined_df: pd.DataFrame, 
    model_names: List[str]
) -> pd.DataFrame:
    """
    Creates a simplified version of the combined metrics with only the key metrics.
    
    Args:
        combined_df: DataFrame with combined metrics
        model_names: List of evaluation model names
        
    Returns:
        DataFrame with simplified metrics
    """
    if combined_df.empty:
        return pd.DataFrame()
    
    # Select key columns: method, parameters, and important metrics
    key_columns = ['method', 'method_id']
    param_columns = [col for col in combined_df.columns if col.startswith('param_')]
    key_columns.extend(param_columns)
    
    # Key metrics to include
    key_metrics = [
        # Direct model metrics (model-specific but not model-prefixed)
        'egalitarian_welfare_perplexity',
        'utilitarian_welfare_perplexity',
        'egalitarian_welfare_cosine',
        'utilitarian_welfare_cosine',
        
        # Agent-specific metrics
        'perplexity_Agent',
        'cosine_similarity_Agent',
        
        # LLM judge metrics
        'min_rank',
        'max_rank',
        'avg_rank',
        'rank_Agent'
    ]
    
    # Find matching columns in the combined DataFrame
    metric_columns = []
    
    # If there are model-specific prefixes, handle them
    for model_name in model_names:
        for metric in key_metrics:
            # Find all columns that contain the metric name with mean/std suffixes
            matching_cols = [
                col for col in combined_df.columns 
                if metric in col and (col.endswith('_mean') or col.endswith('_std'))
            ]
            metric_columns.extend(matching_cols)
            
            # Also check for model-prefixed metrics
            matching_cols = [
                col for col in combined_df.columns 
                if f"{model_name}_{metric}" in col and (col.endswith('_mean') or col.endswith('_std'))
            ]
            metric_columns.extend(matching_cols)
    
    # Direct LLM judge metrics (without model prefix)
    for metric in key_metrics:
        matching_cols = [
            col for col in combined_df.columns 
            if metric in col and (col.endswith('_mean') or col.endswith('_std'))
            and not any(model_name in col for model_name in model_names)
        ]
        metric_columns.extend(matching_cols)
    
    # Agent-specific rank metrics
    agent_rank_cols = [
        col for col in combined_df.columns 
        if col.startswith('rank_Agent') and (col.endswith('_mean') or col.endswith('_std'))
    ]
    metric_columns.extend(agent_rank_cols)
    
    # Combine all columns
    all_columns = key_columns + list(set(metric_columns))
    
    # Create a simplified DataFrame
    simplified_df = combined_df[all_columns].copy()
    
    return simplified_df

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation metrics across seeds"
    )
    parser.add_argument(
        "result_dir",
        help="Path to the result directory (e.g., results/test_concurrency_20250509_112224)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path object
    result_dir = Path(args.result_dir)
    
    if not result_dir.exists():
        logger.error(f"Result directory not found: {result_dir}")
        sys.exit(1)
    
    # Create aggregate directory
    aggregate_dir = create_aggregate_directory(result_dir)
    
    # Find all evaluation model directories
    eval_dir = result_dir / "evaluation"
    
    if not eval_dir.exists():
        logger.error(f"Evaluation directory not found: {eval_dir}")
        sys.exit(1)
    
    model_dirs = [
        d for d in eval_dir.iterdir() 
        if d.is_dir() and d.name != "llm_judge" and d.name != "aggregate"
    ]
    
    logger.info(f"Found {len(model_dirs)} evaluation model directories")
    
    # Aggregate metrics for each model
    eval_dfs = {}
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        logger.info(f"Aggregating metrics for {model_name}")
        
        eval_df = aggregate_model_metrics(result_dir, model_name, aggregate_dir)
        
        if not eval_df.empty:
            eval_dfs[model_name] = eval_df
    
    # Aggregate LLM judge metrics
    llm_judge_df = aggregate_llm_judge_metrics(result_dir, aggregate_dir)
    
    # Combine all metrics
    combined_df = combine_metrics(eval_dfs, llm_judge_df, aggregate_dir)
    
    if not combined_df.empty:
        logger.info(f"Successfully aggregated metrics across {len(eval_dfs)} evaluation models")
        logger.info(f"Combined metrics saved to {aggregate_dir}")
    else:
        logger.warning("No metrics were aggregated")

if __name__ == "__main__":
    main()