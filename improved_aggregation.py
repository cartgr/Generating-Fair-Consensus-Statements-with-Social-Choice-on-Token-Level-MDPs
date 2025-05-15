#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
import re
from typing import Dict, List, Tuple, Optional, Union, Any, Set

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the important parameters that should be included in method identification
IMPORTANT_PARAMETERS = [
    "n", "num_candidates", "num_rounds", 
    "branching_factor", "max_depth", "beam_width"
]

# Define metrics to include in the final aggregated output
METRICS_TO_INCLUDE = {
    "perplexity": {
        "agent_specific": ["perplexity_Agent"],  # Will match any column starting with this
        "aggregate": ["egalitarian_welfare_perplexity", "utilitarian_welfare_perplexity"]
    },
    "cosine": {
        "agent_specific": ["cosine_similarity_Agent"],
        "aggregate": ["egalitarian_welfare_cosine", "utilitarian_welfare_cosine"]
    },
    "rank": {
        "basic": ["min_rank", "max_rank", "avg_rank"],
        "agent_specific": ["rank_Agent"]
    }
}

def create_output_directory(result_dir: Path) -> Path:
    """
    Creates an improved aggregation directory in the evaluation folder.
    
    Args:
        result_dir: Path to the result directory
        
    Returns:
        Path to the created improved aggregation directory
    """
    aggregate_dir = result_dir / "evaluation" / "improved_aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created improved aggregation directory: {aggregate_dir}")
    return aggregate_dir

def normalize_method_name(method_name: str) -> str:
    """
    Normalizes a method name by removing seed information and extra whitespace.

    Args:
        method_name: The method name to normalize

    Returns:
        Normalized method name
    """
    if not method_name or pd.isna(method_name):
        return "unknown"

    # Remove seed information (e.g., "method [seed=42]" -> "method")
    seed_pattern = r'\s*\[seed=\d+\]'
    normalized = re.sub(seed_pattern, '', method_name)

    # Remove any extra whitespace
    normalized = normalized.strip()

    return normalized

def extract_parameters(method_with_params: str) -> Dict[str, Any]:
    """
    Extracts parameters from a method_with_params string.
    
    Args:
        method_with_params: String like "method (param1=value1, param2=value2)"
        
    Returns:
        Dictionary of parameter names and values
    """
    params = {}
    
    # Extract parameter part between parentheses: "method (param1=value1)" -> "param1=value1"
    param_match = re.search(r'\((.*?)\)', method_with_params)
    if param_match:
        param_str = param_match.group(1)
        # Split parameters by comma: "param1=value1, param2=value2" -> ["param1=value1", "param2=value2"]
        param_items = [p.strip() for p in param_str.split(',')]
        
        for item in param_items:
            # Split each parameter by equals sign: "param1=value1" -> ("param1", "value1")
            if '=' in item:
                key, value = item.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to convert value to appropriate type
                try:
                    # Try as int
                    params[key] = int(value)
                except ValueError:
                    try:
                        # Try as float
                        params[key] = float(value)
                    except ValueError:
                        # Keep as string
                        params[key] = value
    
    return params

def create_standardized_method_key(
    method_name: str,
    parameters: Dict[str, Any] = None
) -> Tuple[str, str]:
    """
    Creates a standardized method key with properly formatted parameters.
    
    Args:
        method_name: Base method name
        parameters: Dictionary of parameter names and values
        
    Returns:
        Tuple of (base_method_name, standardized_method_key)
    """
    # Normalize the method name (remove seed info, etc.)
    base_method = normalize_method_name(method_name)
    
    # Filter to only include important parameters
    filtered_params = {}
    if parameters:
        for key, value in parameters.items():
            # Remove param_ prefix if present
            key_name = key.replace("param_", "") if key.startswith("param_") else key
            
            if key_name in IMPORTANT_PARAMETERS and value is not None:
                filtered_params[key_name] = value
    
    # Create method key string with parameters
    if filtered_params:
        # Sort parameters for consistency
        param_items = [f"{k}={v}" for k, v in sorted(filtered_params.items())]
        param_str = ", ".join(param_items)
        method_key = f"{base_method} ({param_str})"
    else:
        method_key = base_method
    
    return base_method, method_key

def collect_evaluation_data(result_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Collects all evaluation data from model-specific directories.
    
    Args:
        result_dir: Path to the result directory
        
    Returns:
        Dictionary mapping model names to their combined evaluation results
    """
    eval_dir = result_dir / "evaluation"
    if not eval_dir.exists():
        logger.error(f"Evaluation directory not found: {eval_dir}")
        return {}
    
    # Find model directories (excluding llm_judge and aggregate)
    model_dirs = [
        d for d in eval_dir.iterdir() 
        if d.is_dir() and d.name not in ["llm_judge", "aggregate", "improved_aggregate"]
    ]
    
    if not model_dirs:
        logger.warning(f"No model evaluation directories found in {eval_dir}")
        return {}
    
    # Collect data for each model
    model_data = {}
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        logger.info(f"Collecting evaluation data for model: {model_name}")
        
        # Find all seed directories
        seed_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]
        
        if not seed_dirs:
            logger.warning(f"No seed directories found for model {model_name}")
            continue
        
        seed_results = []
        
        for seed_dir in seed_dirs:
            eval_file = seed_dir / "evaluation_results.csv"
            
            if not eval_file.exists():
                logger.warning(f"Evaluation results file not found: {eval_file}")
                continue
            
            try:
                # Read the CSV and note which seed directory it came from
                df = pd.read_csv(eval_file)
                
                # Extract seed number from directory name
                seed_match = re.search(r'seed_(\d+)', seed_dir.name)
                seed_num = int(seed_match.group(1)) if seed_match else None
                
                # Add seed directory info if not already in the data
                if seed_num is not None and 'seed_dir' not in df.columns:
                    df['seed_dir'] = seed_num
                
                seed_results.append(df)
                logger.info(f"Read {len(df)} rows from {eval_file}")
            except Exception as e:
                logger.error(f"Error reading {eval_file}: {e}")
        
        if seed_results:
            # Combine all seed results for this model
            model_data[model_name] = pd.concat(seed_results, ignore_index=True)
            logger.info(f"Collected {len(model_data[model_name])} total rows for model {model_name}")
        else:
            logger.warning(f"No valid evaluation results found for model {model_name}")
    
    return model_data

def collect_llm_judge_data(result_dir: Path) -> Optional[pd.DataFrame]:
    """
    Collects all LLM judge ranking data across seeds.
    
    Args:
        result_dir: Path to the result directory
        
    Returns:
        DataFrame with combined ranking data or None if not found
    """
    llm_judge_dir = result_dir / "evaluation" / "llm_judge"
    
    if not llm_judge_dir.exists():
        logger.warning(f"LLM judge directory not found: {llm_judge_dir}")
        return None
    
    # Find all seed directories
    seed_dirs = [d for d in llm_judge_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]
    
    if not seed_dirs:
        logger.warning(f"No seed directories found in {llm_judge_dir}")
        return None
    
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
            
            # Extract seed number from directory name
            seed_match = re.search(r'seed_(\d+)', seed_dir.name)
            seed_num = int(seed_match.group(1)) if seed_match else None
            
            # Add seed directory info if not already in the data
            if seed_num is not None and 'seed_dir' not in df.columns:
                df['seed_dir'] = seed_num
            
            all_rankings.append(df)
            logger.info(f"Read {len(df)} rows from {ranking_file}")
        except Exception as e:
            logger.error(f"Error reading {ranking_file}: {e}")
    
    if not all_rankings:
        logger.warning("No LLM judge ranking results found")
        return None
    
    # Combine all rankings
    combined_df = pd.concat(all_rankings, ignore_index=True)
    logger.info(f"Collected {len(combined_df)} total LLM judge ranking rows")
    
    return combined_df

def process_model_data(model_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    Processes model evaluation data to prepare for aggregation.

    Args:
        model_data: Dictionary mapping model names to their combined evaluation results

    Returns:
        Processed data structure organizing metrics by method key and metric name
    """
    processed_data = {}

    for model_name, df in model_data.items():
        logger.info(f"Processing data for model {model_name}")
        model_processed = {}

        # Process each row
        for _, row in df.iterrows():
            # Get method name and parameters
            method = row.get('method', 'unknown')

            # Check if method_with_params column exists and has information
            method_with_params = None
            if 'method_with_params' in row and pd.notna(row['method_with_params']):
                method_with_params = row['method_with_params']

                # If base method is empty but method_with_params has information, extract it
                if method == 'unknown' or not method or pd.isna(method):
                    # Try to extract base method from method_with_params
                    base_method_match = re.match(r'^([\w-]+)\s*\(', method_with_params)
                    if base_method_match:
                        method = base_method_match.group(1)
                    else:
                        # Just use the whole string as the method name if no parameters found
                        method = method_with_params

            # Collect parameters from param_* columns
            parameters = {}
            for col in row.index:
                if col.startswith('param_') and pd.notna(row[col]):
                    parameters[col] = row[col]

            # If no parameters found in columns but method_with_params exists,
            # try to extract parameters from it
            if not parameters and method_with_params:
                extracted_params = extract_parameters(method_with_params)
                parameters = {f"param_{k}": v for k, v in extracted_params.items()}

            # Create standardized method key
            base_method, method_key = create_standardized_method_key(method, parameters)

            # Store original parameters in the processed data for later use
            if method_key not in model_processed:
                model_processed[method_key] = {
                    'base_method': base_method,
                    'parameters': parameters.copy(),
                    'metrics': {}
                }

            # Extract relevant metrics for this row
            for metric_category, metric_sets in METRICS_TO_INCLUDE.items():
                # Extract agent-specific metrics
                for agent_metric_prefix in metric_sets.get('agent_specific', []):
                    agent_cols = [col for col in row.index if col.startswith(agent_metric_prefix)]

                    for col in agent_cols:
                        if pd.notna(row[col]):
                            # Store the metric value
                            if col not in model_processed[method_key]['metrics']:
                                model_processed[method_key]['metrics'][col] = []

                            model_processed[method_key]['metrics'][col].append(row[col])

                # Extract aggregate metrics
                for metric_name in metric_sets.get('aggregate', []):
                    if metric_name in row and pd.notna(row[metric_name]):
                        if metric_name not in model_processed[method_key]['metrics']:
                            model_processed[method_key]['metrics'][metric_name] = []

                        model_processed[method_key]['metrics'][metric_name].append(row[metric_name])

                # For rank metrics, also extract basic metrics
                if metric_category == 'rank':
                    for basic_metric in metric_sets.get('basic', []):
                        if basic_metric in row and pd.notna(row[basic_metric]):
                            if basic_metric not in model_processed[method_key]['metrics']:
                                model_processed[method_key]['metrics'][basic_metric] = []

                            model_processed[method_key]['metrics'][basic_metric].append(row[basic_metric])

        processed_data[model_name] = model_processed
        logger.info(f"Processed {len(model_processed)} unique method keys for model {model_name}")

    return processed_data

def process_llm_judge_data(llm_judge_df: pd.DataFrame) -> Dict[str, Dict[str, List[float]]]:
    """
    Processes LLM judge ranking data to prepare for aggregation.

    Args:
        llm_judge_df: DataFrame with LLM judge ranking results

    Returns:
        Processed data structure organizing ranking metrics by method key
    """
    if llm_judge_df is None:
        return {}

    processed_rankings = {}

    # Process each row
    for _, row in llm_judge_df.iterrows():
        # Get method name
        method = row.get('method', 'unknown')

        # Check for method_with_params column (common in LLM judge results)
        method_with_params = None
        if 'method_with_params' in row and pd.notna(row['method_with_params']):
            method_with_params = row['method_with_params']

            # If base method is empty but method_with_params has information, extract it
            if method == 'unknown' or not method or pd.isna(method):
                # Try to extract base method from method_with_params
                base_method_match = re.match(r'^([\w-]+)\s*\(', method_with_params)
                if base_method_match:
                    method = base_method_match.group(1)
                else:
                    # Just use the whole string as the method name if no parameters found
                    method = method_with_params

        # Extract parameters from either param_* columns or method_with_params
        parameters = {}

        # First try param_* columns
        for col in row.index:
            if col.startswith('param_') and pd.notna(row[col]):
                parameters[col] = row[col]

        # If method_with_params is available but no parameters extracted from columns,
        # try to extract parameters from method_with_params string
        if not parameters and method_with_params:
            extracted_params = extract_parameters(method_with_params)
            parameters = {f"param_{k}": v for k, v in extracted_params.items()}

        # Create standardized method key
        base_method, method_key = create_standardized_method_key(method, parameters)

        # Initialize data structure for this method key if needed
        if method_key not in processed_rankings:
            processed_rankings[method_key] = {
                'base_method': base_method,
                'parameters': parameters.copy(),
                'metrics': {}
            }

        # Extract rank metrics
        rank_metrics = METRICS_TO_INCLUDE['rank']

        # Extract basic rank metrics
        for metric_name in rank_metrics['basic']:
            if metric_name in row and pd.notna(row[metric_name]):
                if metric_name not in processed_rankings[method_key]['metrics']:
                    processed_rankings[method_key]['metrics'][metric_name] = []

                processed_rankings[method_key]['metrics'][metric_name].append(row[metric_name])

        # Extract agent-specific rank metrics
        for rank_prefix in rank_metrics['agent_specific']:
            agent_cols = [col for col in row.index if col.startswith(rank_prefix)]

            for col in agent_cols:
                if pd.notna(row[col]):
                    if col not in processed_rankings[method_key]['metrics']:
                        processed_rankings[method_key]['metrics'][col] = []

                    processed_rankings[method_key]['metrics'][col].append(row[col])

    logger.info(f"Processed {len(processed_rankings)} unique method keys from LLM judge data")
    return processed_rankings

def aggregate_metrics(
    model_data: Dict[str, Dict[str, Dict[str, List[float]]]],
    llm_judge_data: Dict[str, Dict[str, List[float]]]
) -> pd.DataFrame:
    """
    Aggregates metrics across all models and LLM judge.

    Args:
        model_data: Processed model data by model name, method key, and metric
        llm_judge_data: Processed LLM judge data by method key and metric

    Returns:
        DataFrame with aggregated metrics
    """
    # Collect all unique method keys across all data sources
    all_method_keys = set()

    for model_data_dict in model_data.values():
        all_method_keys.update(model_data_dict.keys())

    all_method_keys.update(llm_judge_data.keys())

    # Prepare data for the aggregated DataFrame
    aggregated_data = []

    for method_key in sorted(all_method_keys):
        # Start with basic method information
        agg_row = {
            "method_with_params": method_key,
        }

        # Get base method name and parameters from any available source
        base_method = None
        parameters = {}

        # Try to get from model data first
        for model_name, model_data_dict in model_data.items():
            if method_key in model_data_dict:
                base_method = model_data_dict[method_key]['base_method']
                if 'parameters' in model_data_dict[method_key]:
                    parameters = model_data_dict[method_key]['parameters']
                break

        # If not found, try LLM judge data
        if base_method is None and method_key in llm_judge_data:
            base_method = llm_judge_data[method_key]['base_method']
            if 'parameters' in llm_judge_data[method_key]:
                parameters = llm_judge_data[method_key]['parameters']

        # Add base method to row
        agg_row["method"] = base_method or "unknown"

        # If parameters dict is empty, try extracting from method key
        if not parameters:
            extracted_params = extract_parameters(method_key)
            parameters = {f"param_{k}": v for k, v in extracted_params.items()}

        # Add parameters as separate columns
        for param_name, param_value in parameters.items():
            # Normalize param name to remove "param_" prefix if present
            clean_param_name = param_name.replace("param_", "")
            agg_row[f"param_{clean_param_name}"] = param_value

        # Aggregate metrics from each model
        for model_name, model_data_dict in model_data.items():
            if method_key in model_data_dict:
                # Get metrics for this method key
                metrics = model_data_dict[method_key]['metrics']

                # Calculate mean and std for each metric
                for metric_name, values in metrics.items():
                    if values:
                        # Add model name prefix for model-specific metrics
                        prefix = f"{model_name}_" if len(model_data) > 1 else ""

                        # Calculate stats only if we have multiple values
                        if len(values) > 1:
                            agg_row[f"{prefix}{metric_name}_mean"] = np.mean(values)
                            agg_row[f"{prefix}{metric_name}_std"] = np.std(values)
                        else:
                            # Single value - just use it as the mean with zero std
                            agg_row[f"{prefix}{metric_name}_mean"] = values[0]
                            agg_row[f"{prefix}{metric_name}_std"] = 0.0

        # Aggregate metrics from LLM judge
        if method_key in llm_judge_data:
            metrics = llm_judge_data[method_key]['metrics']

            # Calculate mean and std for each metric
            for metric_name, values in metrics.items():
                if values:
                    # Calculate stats only if we have multiple values
                    if len(values) > 1:
                        agg_row[f"{metric_name}_mean"] = np.mean(values)
                        agg_row[f"{metric_name}_std"] = np.std(values)
                    else:
                        # Single value - just use it as the mean with zero std
                        agg_row[f"{metric_name}_mean"] = values[0]
                        agg_row[f"{metric_name}_std"] = 0.0

        aggregated_data.append(agg_row)

    # Create DataFrame from aggregated data
    agg_df = pd.DataFrame(aggregated_data)

    return agg_df

def create_formatted_output(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a cleaner, more formatted output DataFrame.

    Args:
        agg_df: Raw aggregated DataFrame

    Returns:
        Formatted DataFrame with cleaned column names and better organization
    """
    # Start with a copy of the input DataFrame
    formatted_df = agg_df.copy()

    # Organize columns:
    column_order = ["method", "method_with_params"]

    # Add parameter columns next
    param_cols = sorted([col for col in formatted_df.columns if col.startswith("param_")])
    column_order.extend(param_cols)

    # Define metric categories and their column ordering
    metric_categories = [
        {
            "name": "perplexity metrics",
            "column_match": ["perplexity"],
            "subcategories": [
                {"name": "egalitarian", "match": ["egalitarian"]},
                {"name": "utilitarian", "match": ["utilitarian"]},
                {"name": "agent-specific", "match": ["Agent"]}
            ]
        },
        {
            "name": "cosine metrics",
            "column_match": ["cosine"],
            "subcategories": [
                {"name": "egalitarian", "match": ["egalitarian"]},
                {"name": "utilitarian", "match": ["utilitarian"]},
                {"name": "agent-specific", "match": ["Agent"]}
            ]
        },
        {
            "name": "rank metrics",
            "column_match": ["rank"],
            "subcategories": [
                {"name": "basic", "match": ["min_rank", "max_rank", "avg_rank"]},
                {"name": "agent-specific", "match": ["rank_Agent"]}
            ]
        }
    ]

    # Add columns according to the defined order
    for category in metric_categories:
        category_cols = []

        # Process models in a specific order for consistency
        model_prefixes = []
        for col in formatted_df.columns:
            parts = col.split('_', 1)
            if len(parts) > 1 and parts[0] not in model_prefixes and parts[0] != 'param':
                model_prefixes.append(parts[0])

        # Sort model prefixes to ensure consistent ordering
        model_prefixes = sorted(model_prefixes)

        # If no model prefixes found, use an empty string for non-prefixed metrics
        if not model_prefixes:
            model_prefixes = ['']

        # Find all columns matching this category, for each model
        for model_prefix in model_prefixes:
            prefix = f"{model_prefix}_" if model_prefix else ""

            for match_str in category["column_match"]:
                # Get matching columns for this model and category
                if model_prefix:
                    matching_cols = [col for col in formatted_df.columns
                                  if match_str in col and col.startswith(model_prefix)]
                else:
                    # For non-prefixed metrics, avoid matching prefixed ones
                    matching_cols = [col for col in formatted_df.columns
                                  if match_str in col and not any(col.startswith(mp) for mp in model_prefixes if mp)]

                # Sort by subcategories
                for subcategory in category["subcategories"]:
                    subcategory_cols = []

                    # Find columns matching this subcategory
                    for submatch in subcategory["match"]:
                        subcategory_match = [col for col in matching_cols if submatch in col]

                        # Pair mean/std columns
                        metric_bases = set()
                        for col in subcategory_match:
                            # Remove _mean and _std suffixes to get the base metric name
                            if col.endswith('_mean'):
                                base = col[:-5]  # Remove '_mean'
                                metric_bases.add(base)
                            elif col.endswith('_std'):
                                base = col[:-4]  # Remove '_std'
                                metric_bases.add(base)

                        # For each base metric name, add mean then std
                        for base in sorted(metric_bases):
                            mean_col = f"{base}_mean"
                            std_col = f"{base}_std"
                            if mean_col in subcategory_match:
                                subcategory_cols.append(mean_col)
                            if std_col in subcategory_match:
                                subcategory_cols.append(std_col)

                    category_cols.extend(subcategory_cols)

        column_order.extend(category_cols)

    # Add any remaining columns not captured above
    remaining_cols = [col for col in formatted_df.columns if col not in column_order]
    column_order.extend(remaining_cols)

    # Reorder columns, but only include columns that actually exist
    valid_columns = [col for col in column_order if col in formatted_df.columns]
    formatted_df = formatted_df[valid_columns]

    return formatted_df

def main(custom_args=None):
    """
    Main function to run the improved aggregation.

    Args:
        custom_args: Optional list of arguments to parse instead of using sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Improved metrics aggregation across seeds"
    )
    parser.add_argument(
        "result_dir",
        help="Path to the result directory (e.g., results/test_concurrency_20250509_112224)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    # Use custom args if provided, otherwise use sys.argv
    args = parser.parse_args(custom_args)

    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Convert to Path object
    result_dir = Path(args.result_dir)

    if not result_dir.exists():
        logger.error(f"Result directory not found: {result_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir = create_output_directory(result_dir)
    
    # Step 1: Collect all evaluation data
    logger.info("Collecting evaluation data from model-specific directories...")
    model_data = collect_evaluation_data(result_dir)
    
    if not model_data:
        logger.error("No evaluation data found. Aborting.")
        sys.exit(1)
    
    # Step 2: Collect LLM judge data
    logger.info("Collecting LLM judge ranking data...")
    llm_judge_df = collect_llm_judge_data(result_dir)
    
    # Step 3: Process collected data
    logger.info("Processing model evaluation data...")
    processed_model_data = process_model_data(model_data)
    
    logger.info("Processing LLM judge ranking data...")
    processed_llm_judge_data = process_llm_judge_data(llm_judge_df) if llm_judge_df is not None else {}
    
    # Step 4: Aggregate metrics
    logger.info("Aggregating metrics across all data sources...")
    aggregated_df = aggregate_metrics(processed_model_data, processed_llm_judge_data)
    
    # Step 5: Create formatted output
    logger.info("Creating formatted output...")
    formatted_df = create_formatted_output(aggregated_df)
    
    # Step 6: Save results
    raw_output_path = output_dir / "aggregated_metrics_raw.csv"
    aggregated_df.to_csv(raw_output_path, index=False)
    logger.info(f"Saved raw aggregated metrics to {raw_output_path}")
    
    formatted_output_path = output_dir / "aggregated_metrics.csv"
    formatted_df.to_csv(formatted_output_path, index=False)
    logger.info(f"Saved formatted aggregated metrics to {formatted_output_path}")
    
    logger.info("Aggregation complete!")

if __name__ == "__main__":
    main()