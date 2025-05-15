#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
from pathlib import Path
import logging
from typing import List, Optional
import pandas as pd
import json

# Set OpenAI logging to WARNING level to suppress debug messages
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

# Import necessary components
from run_experiment import run_experiment_from_config
from src.evaluation import StatementEvaluator

# Default evaluation models
DEFAULT_EVALUATION_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "google/gemma-2-9b-it",
]


def run_experiment_with_evaluation(
    config_path: str,
    skip_llm_judge: bool = False,
    skip_comparative_ranking: bool = False,
    evaluation_models: Optional[List[str]] = None,
    llm_judge_model: str = "o3",
    quiet: bool = False,
):
    """
    Runs an experiment followed by evaluation with multiple models.

    Args:
        config_path: Path to the experiment configuration file
        skip_llm_judge: Whether to skip LLM-as-judge evaluation
        skip_comparative_ranking: Whether to skip comparative statement ranking (enabled by default)
        evaluation_models: Optional list of evaluation models to use (overrides config)
        llm_judge_model: OpenAI model to use for LLM Judge and comparative ranking
        quiet: Whether to suppress detailed output
    """
    # Run the experiment
    print("\n--- PHASE 1: RUNNING EXPERIMENT (STATEMENT GENERATION) ---")
    run_experiment_from_config(config_path)

    # Determine the results directory (the most recently created one)
    config_path_obj = Path(config_path)
    config_dir = config_path_obj.parent

    # Load the config to get default output directory and experiment name
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    output_dir = config.get("output_dir", "results")
    experiment_name = config.get("experiment_name", "experiment")

    # Find the most recent results directory matching the experiment name pattern
    results_dirs = sorted(
        [d for d in Path(output_dir).glob(f"{experiment_name}_*") if d.is_dir()],
        key=os.path.getmtime,
        reverse=True,
    )

    if not results_dirs:
        print(
            f"ERROR: No results directories found in {output_dir} matching {experiment_name}_*"
        )
        sys.exit(1)

    results_dir = results_dirs[0]
    print(f"Using most recent results directory: {results_dir}")

    # Check if the results file exists
    results_file = results_dir / "results.csv"
    config_file = results_dir / "config.yaml"

    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        sys.exit(1)

    # Determine the evaluation models to use
    if evaluation_models is None:
        # Try to get models from config, otherwise use defaults
        with open(config_file, "r") as f:
            exp_config = yaml.safe_load(f)

        config_eval_models = exp_config.get("models", {}).get("evaluation_models")
        single_eval_model = exp_config.get("models", {}).get("evaluation_model")

        if config_eval_models:
            # Config specifies a list of models
            evaluation_models = config_eval_models
        elif single_eval_model:
            # Config specifies a single model
            evaluation_models = [single_eval_model]
        else:
            # Use default models
            print("WARNING: No evaluation models specified in config. Using defaults:")
            for model in DEFAULT_EVALUATION_MODELS:
                print(f"  - {model}")
            evaluation_models = DEFAULT_EVALUATION_MODELS

    # Run LLM judge evaluation once if requested
    llm_judge_results = None
    if not skip_comparative_ranking:
        print("\n--- PHASE 2a: RUNNING LLM JUDGE COMPARISON ---")

        # Create a separate directory for LLM judge results
        llm_judge_dir = results_dir / "evaluation" / "llm_judge"
        llm_judge_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running LLM judge evaluation with model: {llm_judge_model}")
        llm_judge_evaluator = StatementEvaluator(
            evaluation_model="none",  # Not using this for LLM judge only
            llm_judge_model=llm_judge_model,
            include_llm_judge=False,  # Don't run the individual statement evaluation
            include_comparative_ranking=True,  # Only run the comparative ranking
            verbose=not quiet,
        )

        # Run just the LLM judge evaluation
        with open(config_file, "r") as f:
            exp_config = yaml.safe_load(f)

        # Get issue and agent opinions from config
        issue = exp_config.get("scenario", {}).get("issue")
        agent_opinions = exp_config.get("scenario", {}).get("agent_opinions", {})

        # Find all result files for this experiment
        experiment_name = exp_config.get("experiment_name", "experiment")
        output_dir = Path(exp_config.get("output_dir", "results"))
        
        # We'll look for seed data in the current experiment run only
        print("Looking for seed data in the current experiment run...")
        
        # Use only the current results directory - don't search for all experiments with same name
        result_dirs = [results_dir]
        
        # Get the results.csv file from the current experiment
        result_path = results_dir / "results.csv"
        if not result_path.exists():
            print(f"ERROR: Results file not found at {result_path}")
            sys.exit(1)
            
        # Read the results to determine how many unique seeds we have
        try:
            results_df = pd.read_csv(result_path)
            if "seed" in results_df.columns:
                unique_seeds = sorted(results_df["seed"].unique())
                print(f"Found {len(unique_seeds)} unique seeds in the current experiment")
            else:
                print("WARNING: No seed column found in results.csv. Assuming single seed.")
                unique_seeds = [None]
        except Exception as e:
            print(f"ERROR: Failed to read results file: {e}")
            sys.exit(1)
        
        # Process each seed separately
        for seed_idx, seed_value in enumerate(unique_seeds):
            # Create a seed-specific directory for LLM judge results
            seed_dir = llm_judge_dir / f"seed_{seed_idx}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Filter results for this seed
                if seed_value is not None:
                    seed_results_df = results_df[results_df["seed"] == seed_value].copy()
                    print(f"\nProcessing results for seed {seed_value}")
                else:
                    # If no seed column or only one seed, use all results
                    seed_results_df = results_df.copy()
                    seed_value = "default_seed"
                    print("\nProcessing all results (no seed specified)")
                
                # Collect statements for this seed only
                statements = {}
                for _, row in seed_results_df.iterrows():
                    method = row.get("method", "Unknown method")
                    statement = row.get("statement", "No statement generated")
                    
                    # Skip entries with errors
                    if statement == "ERROR" or pd.isna(statement):
                        continue
                    
                    # Get parameter columns
                    param_cols = [col for col in row.index if col.startswith("param_")]

                    # Create a dictionary of parameter values
                    params_dict = {}
                    for param in param_cols:
                        if pd.notna(row[param]):
                            params_dict[param] = row[param]

                    # Use the standardized method identifier function
                    from src.utils import create_method_identifier
                    method_key = create_method_identifier(
                        method_name=method,
                        params_dict=params_dict,
                        include_seed=False  # We're already processing by seed
                    )
                    
                    # Store the statement with the enhanced method key
                    statements[method_key] = statement
                
                print(f"Collected {len(statements)} statements for seed {seed_value}")
                
                # Run the comparative ranking for this seed
                if statements and issue and agent_opinions:
                    try:
                        print(f"\nRunning comparative ranking for seed {seed_value}...")
                        seed_llm_judge_results = llm_judge_evaluator.evaluate_comparative_rankings(
                            statements,
                            issue,
                            agent_opinions,
                        )
                        
                        # Save the ranking matrix as a CSV file
                        ranking_matrix = seed_llm_judge_results.get("ranking_matrix", {})
                        
                        # Create a DataFrame for the rankings
                        ranking_df = pd.DataFrame()
                        
                        # Add a row for each method
                        methods = list(seed_llm_judge_results.get("method_min_ranks", {}).keys())
                        
                        for method_name in methods:
                            # Extract base method name and parameters for preservation
                            base_method = method_name
                            param_dict = {}
                            
                            # Parse parameters from method name if it contains them (e.g., "best_of_n (n=10)")
                            if " (" in method_name:
                                parts = method_name.split(" (", 1)
                                base_method = parts[0]
                                
                                if parts[1].endswith(")"):
                                    param_str = parts[1].rstrip(")")
                                    
                                    # Parse parameters into a dictionary
                                    for param_item in param_str.split(", "):
                                        if "=" in param_item:
                                            param_name, param_value = param_item.split("=", 1)
                                            # Try to convert to numeric if possible
                                            try:
                                                param_value = float(param_value)
                                                # Convert to int if it's a whole number
                                                if param_value.is_integer():
                                                    param_value = int(param_value)
                                            except ValueError:
                                                # Keep as string if not numeric
                                                pass
                                            
                                            param_dict[f"param_{param_name}"] = param_value
                            
                            row_data = {
                                "method": base_method,  # Use base method name without parameters
                                "seed": seed_value,
                                # Add the original method name with parameters for reference
                                "method_with_params": method_name
                            }
                            
                            # Add all extracted parameters to the row data
                            row_data.update(param_dict)
                            
                            # Add the min, max, and avg ranks
                            row_data["min_rank"] = seed_llm_judge_results.get("method_min_ranks", {}).get(method_name)
                            row_data["max_rank"] = seed_llm_judge_results.get("method_max_ranks", {}).get(method_name)
                            row_data["avg_rank"] = seed_llm_judge_results.get("method_avg_ranks", {}).get(method_name)
                            
                            # Add the rank for each agent
                            for agent_id, agent_data in ranking_matrix.items():
                                method_rankings = agent_data.get("method_ranking", {})
                                rank = method_rankings.get(method_name)
                                row_data[f"rank_{agent_id}"] = rank
                            
                            # Add welfare indicator
                            if method_name == seed_llm_judge_results.get("maximin_welfare_method"):
                                row_data["is_maximin_best"] = 1
                            else:
                                row_data["is_maximin_best"] = 0
                                
                            if method_name == seed_llm_judge_results.get("utilitarian_welfare_method"):
                                row_data["is_utilitarian_best"] = 1
                            else:
                                row_data["is_utilitarian_best"] = 0
                            
                            # Add to DataFrame
                            ranking_df = pd.concat([ranking_df, pd.DataFrame([row_data])], ignore_index=True)
                        
                        # Save to CSV
                        ranking_csv_path = seed_dir / "ranking_results.csv"
                        ranking_df.to_csv(ranking_csv_path, index=False)
                        
                        # Also save the agent reasoning to a separate CSV
                        reasoning_df = pd.DataFrame()
                        
                        for agent_id, agent_data in ranking_matrix.items():
                            reasoning = agent_data.get("reasoning", "No reasoning provided")
                            reasoning_df = pd.concat([
                                reasoning_df, 
                                pd.DataFrame([{
                                    "seed": seed_value,
                                    "agent": agent_id, 
                                    "reasoning": reasoning
                                }])
                            ], ignore_index=True)
                        
                        reasoning_csv_path = seed_dir / "ranking_reasoning.csv"
                        reasoning_df.to_csv(reasoning_csv_path, index=False)
                        
                        # Also save a JSON version for backward compatibility
                        ranking_matrix_path = seed_dir / "comparative_ranking_matrix.json"
                        with open(ranking_matrix_path, "w") as f:
                            json.dump(ranking_matrix, f, indent=2)
                        
                        print(f"LLM judge results for seed {seed_value} saved to {seed_dir}")
                        
                        # Store the first seed's results in the variable (but don't save to parent folder)
                        if seed_idx == 0:
                            llm_judge_results = seed_llm_judge_results
                            
                    except Exception as e:
                        print(f"ERROR in LLM judge evaluation for seed {seed_value}: {e}")
                        import traceback
                        print(traceback.format_exc())
                else:
                    print(f"Skipping seed {seed_value}: No valid statements, issue, or agent opinions")
                    
            except Exception as e:
                print(f"Error processing {result_path}: {e}")
                import traceback
                print(traceback.format_exc())

    # Run regular evaluation on the results for each model, processing each seed separately
    print("\n--- PHASE 2b: RUNNING STANDARD EVALUATIONS ---")

    for eval_model in evaluation_models:
        model_name_safe = eval_model.replace("/", "_")
        print(f"\nEvaluating with model: {eval_model}")

        evaluator = StatementEvaluator(
            evaluation_model=eval_model,
            llm_judge_model=None,  # No need for LLM judge in standard evaluation
            include_llm_judge=False,  # Skip LLM judge evaluation (already done)
            include_comparative_ranking=False,  # Skip comparative ranking (already done)
            verbose=not quiet,
        )

        # Process each seed separately for standard evaluation
        for seed_idx, seed_value in enumerate(unique_seeds):
            # Create seed-specific evaluation directory
            seed_eval_dir = results_dir / "evaluation" / model_name_safe / f"seed_{seed_idx}"
            seed_eval_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Create a temporary CSV file with just this seed's results
                if seed_value is not None:
                    seed_results_df = results_df[results_df["seed"] == seed_value].copy()
                else:
                    # If no seed column or only one seed, use all results
                    seed_results_df = results_df.copy()
                    seed_value = "default_seed"
                    
                temp_result_path = seed_eval_dir / "temp_results.csv"
                seed_results_df.to_csv(temp_result_path, index=False)
                    
                print(f"\nEvaluating seed {seed_value} results with {eval_model}")
                    
                # Run evaluation for this seed (with is_seed_specific=True to avoid parent folder duplication)
                evaluator.evaluate_results_file(
                    results_path=str(temp_result_path),
                    config_path=str(config_file),
                    output_dir=str(seed_eval_dir),
                    is_seed_specific=True,
                )
                
                # Clean up temporary file
                if temp_result_path.exists():
                    os.remove(temp_result_path)
                
                # No longer creating symlinks or copying results to parent directory
                # All results are kept only in seed-specific folders
                pass
            except Exception as e:
                print(f"Error evaluating seed {seed_idx} with {eval_model}: {e}")
                import traceback
                print(traceback.format_exc())

    print("\n--- PHASE 3: AGGREGATING EVALUATION RESULTS ---")
    try:
        # First try the improved aggregation script
        try:
            # Import the improved aggregation script
            import sys
            import importlib.util

            # Try to find the improved_aggregation.py module
            improved_aggregation_path = Path(__file__).parent / "improved_aggregation.py"
            if improved_aggregation_path.exists():
                # Module exists, import and run it
                print(f"Found improved aggregation script at {improved_aggregation_path}")
                sys.path.insert(0, str(Path(__file__).parent))
                import improved_aggregation
                print(f"Running improved aggregation script on {results_dir}")
                # Call the main function, but pass the path as a list for argparse compatibility
                improved_aggregation.main([str(results_dir)])
                print(f"Improved aggregation completed successfully")
            else:
                # Fall back to direct approach
                raise ImportError("improved_aggregation script not found")

        except (ImportError, AttributeError) as e:
            print(f"Improved aggregation not available: {e}")
            print("Falling back to basic aggregation...")

            # Import the basic aggregation script
            from aggregate_evaluation import create_aggregate_directory, aggregate_model_metrics
            from aggregate_evaluation import aggregate_llm_judge_metrics, combine_metrics

            # Create aggregate directory
            aggregate_dir = create_aggregate_directory(results_dir)

            # Get the list of model directories
            eval_dir = results_dir / "evaluation"
            model_dirs = [
                d for d in eval_dir.iterdir()
                if d.is_dir() and d.name != "llm_judge" and d.name != "aggregate"
            ]

            # Aggregate metrics for each model
            eval_dfs = {}
            for model_dir in model_dirs:
                model_name = model_dir.name
                print(f"Aggregating metrics for {model_name}")
                eval_df = aggregate_model_metrics(results_dir, model_name, aggregate_dir)
                if not eval_df.empty:
                    eval_dfs[model_name] = eval_df

            # Aggregate LLM judge metrics
            llm_judge_df = aggregate_llm_judge_metrics(results_dir, aggregate_dir)

            # Combine all metrics
            combined_df = combine_metrics(eval_dfs, llm_judge_df, aggregate_dir)

            if not combined_df.empty:
                print(f"Successfully aggregated metrics across seeds")
                print(f"Combined metrics saved to {aggregate_dir}")
            else:
                print("Warning: No metrics were aggregated")
    except Exception as e:
        print(f"Error aggregating evaluation results: {e}")
        import traceback
        print(traceback.format_exc())

    print("\n--- EXPERIMENT AND EVALUATION COMPLETE ---")
    print(f"Results and evaluation available in: {results_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment with statement generation and evaluation"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/experiment_config.yaml",
        help="Path to the experiment configuration YAML file (default: configs/experiment_config.yaml)",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip the LLM-as-judge evaluation step",
    )
    parser.add_argument(
        "--skip-comparative-ranking",
        action="store_true",
        help="Skip the comparative ranking of statements for egalitarian welfare (enabled by default)",
    )
    parser.add_argument(
        "--llm-judge-model",
        type=str,
        default="o3",
        help="OpenAI model to use for LLM-as-judge and comparative ranking evaluation",
    )
    parser.add_argument(
        "--evaluation-models",
        type=str,
        nargs="+",
        help="Override the evaluation models specified in the config (space-separated list)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")

    args = parser.parse_args()

    # Run the experiment with evaluation
    run_experiment_with_evaluation(
        config_path=args.config,
        skip_llm_judge=args.skip_llm_judge,
        skip_comparative_ranking=args.skip_comparative_ranking,
        evaluation_models=args.evaluation_models,
        llm_judge_model=args.llm_judge_model,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
