#!/usr/bin/env python3
import argparse
import yaml
import json
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from typing import List
import logging

# Set OpenAI logging to WARNING level to suppress debug messages
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

# Import the enhanced evaluation module
from src.evaluation import StatementEvaluator

# Default evaluation models
DEFAULT_EVALUATION_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "google/gemma-2-9b-it"
]

def main():
    """
    Command-line interface for post-hoc evaluation of statements.
    This script is a thin wrapper around the enhanced evaluation module,
    maintaining backward compatibility with the original post_hoc_evaluate.py.
    """
    parser = argparse.ArgumentParser(
        description="Post-hoc evaluation of consensus statements"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML config file containing issue and agent opinions",
    )
    parser.add_argument("--issue", type=str, help="The issue being discussed")
    parser.add_argument(
        "--statements",
        type=str,
        help="Path to a JSON or YAML file with method:statement pairs",
    )
    parser.add_argument(
        "--statements-text",
        type=str,
        help="Path to a text file with method and statement pairs (Method: ...\nStatement: ...)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Path to an existing results directory to re-evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results. If not provided and --results-dir is used, results will be saved in a subdirectory of the results-dir",
    )
    parser.add_argument(
        "--models", 
        type=str, 
        nargs="+",
        default=DEFAULT_EVALUATION_MODELS,
        help=f"Evaluation models to use (space-separated list). Default: {' '.join(DEFAULT_EVALUATION_MODELS)}"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Model to use for embeddings",
    )
    parser.add_argument(
        "--llm-judge-model",
        type=str,
        default="o3",
        help="Model to use for LLM-as-judge evaluation (requires OpenAI API key in .env)",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip the LLM-as-judge evaluation step.",
    )
    parser.add_argument(
        "--skip-comparative-ranking",
        action="store_true",
        help="Skip the comparative ranking of statements for egalitarian welfare (enabled by default).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )
    # For backwards compatibility
    parser.add_argument(
        "--model", 
        type=str,
        help="DEPRECATED: Use --models instead. Single evaluation model to use."
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.results_dir and not args.config and not args.statements and not args.statements_text:
        parser.error(
            "You must provide either --results-dir, --config, --statements, or --statements-text"
        )

    # Handle evaluation models
    evaluation_models = args.models
    
    # For backward compatibility, if --model is specified, use it instead
    if args.model:
        print("WARNING: --model is deprecated. Use --models instead for multiple model support.")
        evaluation_models = [args.model]

    # Check if we're using an existing results directory
    if args.results_dir:
        results_path = Path(args.results_dir) / "results.csv"
        config_path = Path(args.results_dir) / "config.yaml"
        
        if not results_path.exists():
            parser.error(f"Results file not found: {results_path}")
        
        # Run LLM judge evaluation for each seed if requested
        if not args.skip_comparative_ranking:
            print("\n--- Running LLM judge comparison evaluation ---")
            
            # Create a separate directory for LLM judge results
            if args.output_dir:
                llm_judge_dir = Path(args.output_dir) / "llm_judge"
            else:
                llm_judge_dir = Path(args.results_dir) / "evaluation" / "llm_judge"
            
            llm_judge_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Running LLM judge evaluation with model: {args.llm_judge_model}")
            
            # Initialize the evaluator for LLM judge only
            llm_judge_evaluator = StatementEvaluator(
                evaluation_model="none",  # Not using this for LLM judge only
                llm_judge_model=args.llm_judge_model,
                include_llm_judge=False,  # Don't run the individual statement evaluation
                include_comparative_ranking=True,  # Only run the comparative ranking
                embedding_model=args.embedding_model,
                verbose=not args.quiet
            )
            
            # Load config to get issue and agent opinions
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
            elif args.config:
                with open(args.config, "r") as f:
                    config = yaml.safe_load(f)
            else:
                print("ERROR: Could not find config file.")
                return
                
            # Get issue and agent opinions
            issue = config.get("scenario", {}).get("issue")
            agent_opinions = config.get("scenario", {}).get("agent_opinions", {})
            
            # Get the experiment name and directory
            experiment_name = config.get("experiment_name", "experiment")
            output_dir = Path(config.get("output_dir", "results"))
            
            # We'll look for seed data in the current experiment run only
            print("Looking for seed data in the current experiment run...")
            
            # Use only the specified results directory
            result_path = Path(args.results_dir) / "results.csv"
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
                print(f"\n--- Processing seed {seed_idx} (value={seed_value}) ---")
                
                # Create a seed-specific directory
                seed_dir = llm_judge_dir / f"seed_{seed_idx}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                
                # Filter results for this seed only
                if seed_value is not None:
                    seed_results = results_df[results_df["seed"] == seed_value].copy()
                    print(f"Found {len(seed_results)} results for seed {seed_value}")
                else:
                    # If no seed column, use all results
                    seed_results = results_df.copy()
                    print(f"Using all {len(seed_results)} results (no seed column)")
                
                # Collect statements for this seed only
                statements = {}
                
                # Process each row in the filtered results
                for _, row in seed_results.iterrows():
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
                
                print(f"Collected {len(statements)} valid statements for seed {seed_value}")
                
                # Run the comparative ranking for this seed
                if statements and issue and agent_opinions:
                    try:
                        print(f"Running LLM judge comparative ranking for seed {seed_value}...")
                        llm_judge_results = llm_judge_evaluator.evaluate_comparative_rankings(
                            statements,
                            issue,
                            agent_opinions,
                        )
                        
                        # Save the ranking matrix as a CSV file
                        ranking_matrix = llm_judge_results.get("ranking_matrix", {})
                        
                        # Create a DataFrame for the rankings
                        ranking_df = pd.DataFrame()
                        
                        # Add a row for each method
                        methods = list(llm_judge_results.get("method_min_ranks", {}).keys())
                        
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
                            row_data["min_rank"] = llm_judge_results.get("method_min_ranks", {}).get(method_name)
                            row_data["max_rank"] = llm_judge_results.get("method_max_ranks", {}).get(method_name)
                            row_data["avg_rank"] = llm_judge_results.get("method_avg_ranks", {}).get(method_name)
                            
                            # Add the rank for each agent
                            for agent_id, agent_data in ranking_matrix.items():
                                method_rankings = agent_data.get("method_ranking", {})
                                rank = method_rankings.get(method_name)
                                row_data[f"rank_{agent_id}"] = rank
                            
                            # Add welfare indicator
                            if method_name == llm_judge_results.get("maximin_welfare_method"):
                                row_data["is_maximin_best"] = 1
                            else:
                                row_data["is_maximin_best"] = 0
                                
                            if method_name == llm_judge_results.get("utilitarian_welfare_method"):
                                row_data["is_utilitarian_best"] = 1
                            else:
                                row_data["is_utilitarian_best"] = 0
                            
                            # Add to DataFrame
                            ranking_df = pd.concat([ranking_df, pd.DataFrame([row_data])], ignore_index=True)
                        
                        # Save to CSV in the seed-specific directory
                        ranking_csv_path = seed_dir / "ranking_results.csv"
                        ranking_df.to_csv(ranking_csv_path, index=False)
                        
                        # Also save the agent reasoning to a separate CSV
                        reasoning_df = pd.DataFrame()
                        
                        for agent_id, agent_data in ranking_matrix.items():
                            reasoning = agent_data.get("reasoning", "No reasoning provided")
                            reasoning_df = pd.concat([
                                reasoning_df, 
                                pd.DataFrame([{
                                    "agent": agent_id,
                                    "seed": seed_value,
                                    "reasoning": reasoning
                                }])
                            ], ignore_index=True)
                        
                        reasoning_csv_path = seed_dir / "ranking_reasoning.csv"
                        reasoning_df.to_csv(reasoning_csv_path, index=False)
                        
                        # Save a JSON version with the full ranking data
                        ranking_matrix_path = seed_dir / "comparative_ranking_matrix.json"
                        with open(ranking_matrix_path, "w") as f:
                            json.dump(ranking_matrix, f, indent=2)
                            
                        print(f"LLM judge results for seed {seed_value} saved to {seed_dir}")
                    except Exception as e:
                        print(f"ERROR in LLM judge evaluation for seed {seed_value}: {e}")
                        import traceback
                        print(traceback.format_exc())
                else:
                    print(f"Skipping seed {seed_value}: No valid statements, issue, or agent opinions")
        
        # Run standard evaluation with each model
        print("\n--- Running standard model evaluations ---")
        for eval_model in evaluation_models:
            print(f"\nEvaluating with model: {eval_model}")
            model_name_safe = eval_model.replace("/", "_")

            # Initialize the evaluator for this model (without LLM judge)
            evaluator = StatementEvaluator(
                evaluation_model=eval_model,
                llm_judge_model=None,  # Not using LLM judge in standard evaluation
                include_llm_judge=False,  # Skip LLM judge evaluation (already done separately)
                include_comparative_ranking=False,  # Skip comparative ranking (already done separately)
                embedding_model=args.embedding_model,
                verbose=not args.quiet
            )

            # Process each seed separately
            print(f"Processing {len(unique_seeds)} seeds separately...")

            for seed_idx, seed_value in enumerate(unique_seeds):
                print(f"\n--- Processing seed {seed_idx} (value={seed_value}) ---")

                # Filter results for this seed only
                if seed_value is not None:
                    seed_results = results_df[results_df["seed"] == seed_value].copy()
                    print(f"Found {len(seed_results)} results for seed {seed_value}")
                else:
                    # If no seed column, use all results
                    seed_results = results_df.copy()
                    print(f"Using all {len(seed_results)} results (no seed column)")

                # Create a temporary results file for this seed
                temp_results_path = Path(args.results_dir) / f"temp_seed_{seed_idx}_results.csv"
                seed_results.to_csv(temp_results_path, index=False)

                # Create a seed-specific directory
                if args.output_dir:
                    seed_output_dir = Path(args.output_dir) / model_name_safe / f"seed_{seed_idx}"
                else:
                    # Create a subdirectory in the results directory
                    seed_output_dir = Path(args.results_dir) / "evaluation" / model_name_safe / f"seed_{seed_idx}"

                seed_output_dir.mkdir(parents=True, exist_ok=True)

                # Use the evaluator to process the seed-specific results file
                evaluator.evaluate_results_file(
                    results_path=str(temp_results_path),
                    config_path=str(config_path) if config_path.exists() else args.config,
                    output_dir=str(seed_output_dir),
                    is_seed_specific=True
                )

                # Clean up temporary file
                if temp_results_path.exists():
                    os.remove(temp_results_path)
        
        print("\n--- Aggregating evaluation results across seeds ---")
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
                    print(f"Running improved aggregation script on {args.results_dir}")
                    # Call the main function, but pass the path as a list for argparse compatibility
                    improved_aggregation.main([str(args.results_dir)])
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
                results_dir = Path(args.results_dir)
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

        print("\nEvaluation complete for all models.")
        return

    # If we're not using an existing results directory, we need statements and config
    issue = None
    agent_opinions = {}
    statements = {}

    # Load config
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        if config:
            issue = config.get("scenario", {}).get("issue")
            agent_opinions = config.get("scenario", {}).get("agent_opinions", {})
    elif args.issue:
        issue = args.issue
    
    if not issue or not agent_opinions:
        parser.error(
            "Could not extract issue and agent opinions. "
            "Please provide a valid config file with 'scenario' section."
        )

    # Load statements
    if args.statements:
        # Check file extension
        if args.statements.endswith(".json"):
            with open(args.statements, "r") as f:
                statements = json.load(f)
        elif args.statements.endswith(".yaml") or args.statements.endswith(".yml"):
            with open(args.statements, "r") as f:
                statements = yaml.safe_load(f)
        else:
            parser.error("Statements file must be JSON or YAML")

    elif args.statements_text:
        # Parse text file with method and statement pairs
        method = None
        statement_lines = []

        with open(args.statements_text, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Method:"):
                    # Save previous method and statement if they exist
                    if method and statement_lines:
                        statements[method] = "\n".join(statement_lines)

                    # Start new method
                    method = line[len("Method:") :].strip()
                    statement_lines = []
                elif line.startswith("Statement:"):
                    statement_lines = [line[len("Statement:") :].strip()]
                elif method and len(statement_lines) > 0:
                    # Continue previous statement
                    statement_lines.append(line)

        # Save the last method and statement
        if method and statement_lines:
            statements[method] = "\n".join(statement_lines)

    if not statements:
        parser.error(
            "No statements found to evaluate. Use --statements or --statements-text."
        )

    # Process with each evaluation model
    for eval_model in evaluation_models:
        print(f"\nEvaluating with model: {eval_model}")
        model_name_safe = eval_model.replace("/", "_")
        
        # Initialize the evaluator for this model
        evaluator = StatementEvaluator(
            evaluation_model=eval_model,
            llm_judge_model=args.llm_judge_model if (not args.skip_llm_judge or not args.skip_comparative_ranking) else None,
            include_llm_judge=not args.skip_llm_judge,
            include_comparative_ranking=not args.skip_comparative_ranking,
            embedding_model=args.embedding_model,
            verbose=not args.quiet
        )
        
        # Run evaluation
        results_df = evaluator.evaluate_statements(
            statements=statements,
            issue=issue,
            agent_opinions=agent_opinions,
        )

        # Save results if output directory specified
        if args.output_dir:
            base_output_dir = Path(args.output_dir)
            # Create a model-specific subdirectory
            output_dir = base_output_dir / model_name_safe
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            results_path = output_dir / "evaluation_results.csv"
            results_df.to_csv(results_path, index=False)
            
            if not args.quiet:
                print(f"Results saved to {results_path}")
            
            # Save config
            eval_config = {
                "scenario": {
                    "issue": issue,
                    "agent_opinions": agent_opinions,
                },
                "evaluation": {
                    "evaluation_model": eval_model,
                    "embedding_model": args.embedding_model,
                    "include_llm_judge": not args.skip_llm_judge,
                    "llm_judge_model": args.llm_judge_model if not args.skip_llm_judge else None,
                },
                "statements": statements,
            }
            
            config_path = output_dir / "evaluation_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(eval_config, f, default_flow_style=False)
            
            if not args.quiet:
                print(f"Evaluation config saved to {config_path}")
    
    # Skip aggregation when using direct statements rather than an existing results directory
    print("\nEvaluation complete for all models.")

if __name__ == "__main__":
    main()