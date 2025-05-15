import yaml
import datetime
import pandas as pd
from pathlib import Path
import traceback  # For detailed error logging
import time
import itertools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union, Dict, Any

# Import necessary components from other modules within the src package
from .methods import get_method_generator

# Default evaluation models
DEFAULT_EVALUATION_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "google/gemma-2-9b-it"
]

# Thread-local storage for per-thread data
thread_local = threading.local()

# Rate limiter for API calls
class APIRateLimiter:
    """
    A token bucket rate limiter for API calls.
    Ensures API calls are limited to a specified rate to avoid throttling.
    """
    def __init__(self, rate=5, per=1.0):
        """
        Initialize the rate limiter.
        
        Args:
            rate: Maximum number of requests per time period
            per: Time period in seconds
        """
        self.rate = rate
        self.per = per
        self.tokens = rate
        self.last_check = time.time()
        self.lock = threading.RLock()
        
    def wait_for_token(self):
        """
        Wait for an available token before proceeding.
        This method blocks until a token is available.
        """
        with self.lock:
            now = time.time()
            time_passed = now - self.last_check
            self.last_check = now
            self.tokens += time_passed * (self.rate / self.per)
            self.tokens = min(self.tokens, self.rate)
            
            if self.tokens < 1:
                sleep_time = (1 - self.tokens) * self.per / self.rate
                time.sleep(sleep_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class Experiment:
    """
    Encapsulates the logic for running one experiment configuration.
    Loads configuration, sets up output, iterates through methods,
    runs generation and evaluation, and saves results.
    """

    def __init__(self, config: dict):
        """
        Initializes the experiment with a configuration dictionary.

        Args:
            config: The experiment configuration loaded from a YAML file.
        """
        self.config = config
        self.base_seed = config.get("seed")  # Store the main seed
        self.num_seeds = config.get("num_seeds", 1)  # Default to 1 if not specified
        # Extract common parameters from config for convenience
        self.issue = config["scenario"]["issue"]
        self.agent_opinions = config["scenario"]["agent_opinions"]
        self.generation_model_id = config["models"]["generation_model"]
        
        # Handle evaluation models (support both singular and plural forms)
        if "evaluation_models" in config["models"]:
            self.evaluation_model_ids = config["models"]["evaluation_models"]
        elif "evaluation_model" in config["models"]:
            # For backward compatibility
            self.evaluation_model_ids = [config["models"]["evaluation_model"]]
        else:
            # Use default models if none specified
            print("WARNING: No evaluation models specified in config. Using defaults:")
            for model in DEFAULT_EVALUATION_MODELS:
                print(f"  - {model}")
            self.evaluation_model_ids = DEFAULT_EVALUATION_MODELS
            # Update the config for saving
            config["models"]["evaluation_models"] = DEFAULT_EVALUATION_MODELS
        
        self.methods_to_run = config["methods_to_run"]

        # Concurrency settings
        self.concurrent_execution = config.get("concurrent_execution", True)
        self.max_concurrent_methods = config.get("max_concurrent_methods", 4)
        self.api_rate_limit = config.get("api_rate_limit", 5)  # Requests per second
        
        # Create a shared rate limiter for API calls
        self.rate_limiter = APIRateLimiter(rate=self.api_rate_limit, per=1.0)
        
        # Print concurrency settings
        print(f"Concurrent execution: {'Enabled' if self.concurrent_execution else 'Disabled'}")
        if self.concurrent_execution:
            print(f"Maximum concurrent methods: {self.max_concurrent_methods}")
            print(f"API rate limit: {self.api_rate_limit} requests per second")

        # Create a unique output directory for this run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config.get('experiment_name', 'experiment')}_{timestamp}"
        # Use Path object for robust path handling
        self.run_output_dir = Path(config.get("output_dir", "results")) / run_name
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {self.run_output_dir}")

        # Save the config used for this run for reproducibility
        self.config_path = self.run_output_dir / "config.yaml"
        try:
            with open(self.config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"Saved configuration to {self.config_path}")
        except Exception as e:
            print(f"Warning: Could not save config file to {self.config_path}: {e}")

    def _run_method_with_config(self, method_name: str, method_config: Dict[str, Any], 
                              current_seed: Optional[int], run_label: str) -> Dict[str, Any]:
        """
        Run a single method with the specified configuration.
        
        Args:
            method_name: The name of the method to run
            method_config: The configuration for this method
            current_seed: The seed to use for this run
            run_label: A label for this run for logging purposes
            
        Returns:
            A dictionary containing the run data
        """
        # Initialize thread-local rate limiter if not already present
        if not hasattr(thread_local, 'rate_limiter'):
            thread_local.rate_limiter = self.rate_limiter
            
        run_data = {
            "method": method_name,
            "issue": self.issue,
            "statement": None,  # Default value
            "config_file": str(self.config_path),
            "error_message": None,
            "seed": current_seed,  # Add the seed to the results
            # Add *specific* method parameters for this run to results
            **{f"param_{k}": v for k, v in method_config.items()},
        }

        try:
            # 1. Instantiate the generator for the current method & config
            generator = get_method_generator(
                method_name, method_config, self.generation_model_id
            )

            # 2. Generate statement (with rate limiting)
            thread_local.rate_limiter.wait_for_token()

            print(f"\n--- Running: {run_label} with seed {current_seed} ---")
            start_time = time.time()
            statement = generator.generate_statement(
                self.issue, self.agent_opinions
            )
            generation_time = time.time() - start_time
            print(f"  Generated Statement: {statement}")
            run_data["statement"] = statement
            run_data["generation_time_s"] = generation_time

            # Save pre-brushup statement if available
            if hasattr(generator, 'pre_brushup_statement'):
                pre_brushup = generator.pre_brushup_statement
                run_data["pre_brushup_statement"] = pre_brushup
                if pre_brushup != statement:
                    print(f"  Pre-brushup Statement: {pre_brushup}")

            # Evaluation has been moved to post_hoc_evaluate.py
            # Set a flag to indicate that this result needs evaluation
            run_data["evaluation_status"] = "pending"

        except Exception as e:
            error_msg = f"ERROR running {run_label}: {e}"
            print(error_msg)
            print(traceback.format_exc())  # Print full traceback for debugging
            run_data["statement"] = "ERROR"
            run_data["error_message"] = str(e)

        return run_data
        
    def run(self) -> pd.DataFrame | None:
        """
        Runs the experiment for all methods specified in the configuration.
        If method parameters are specified as lists, runs all combinations.
        If num_seeds > 1, runs each configuration multiple times with different seeds.
        
        If concurrent_execution is enabled, methods will be run in parallel.

        Instantiates generators, calls generation and evaluation,
        collects results, and saves them to a CSV file.

        Returns:
            A pandas DataFrame containing the results, or None if errors occurred.
        """
        results_list = []  # Store results for each method run/parameter combination

        print("\n--- Starting Experiment Run ---")
        print(f"Number of seeds: {self.num_seeds}")
        print(f"Evaluation models configured: {', '.join(self.evaluation_model_ids)}")
        
        # Iterate over seeds
        for seed_idx in range(self.num_seeds):
            current_seed = self.base_seed + seed_idx if self.base_seed is not None else None
            print(f"\n--- Running with seed: {current_seed} ({seed_idx+1}/{self.num_seeds}) ---")
            
            # Collect all method configs for this seed
            all_method_configs = []
            
            # First, prepare all method configurations for this seed
            for method_name in self.methods_to_run:
                print(f"\n--- Preparing Method: {method_name} ---")
                # Get method-specific config section (e.g., config['mcts'])
                # Provide an empty dict {} if no specific config exists for the method
                base_method_config = self.config.get(method_name, {})
                # Inject the current seed into the method's config
                if current_seed is not None:
                    base_method_config["seed"] = current_seed

                # --- Parameter Variation Handling ---
                # Separate parameters with list values (variations) from others
                list_params = {
                    k: v for k, v in base_method_config.items() if isinstance(v, list)
                }
                single_value_params = {
                    k: v for k, v in base_method_config.items() if not isinstance(v, list)
                }

                run_configs = []
                if not list_params:
                    # No variations, run once with the original config
                    run_configs.append(base_method_config.copy())
                    print(f"  Running {method_name} with parameters: {base_method_config}")
                else:
                    # Generate all combinations of list parameters
                    param_names = list(list_params.keys())
                    value_combinations = list(itertools.product(*list_params.values()))
                    print(
                        f"  Detected parameter variations for {method_name}: {param_names}"
                    )
                    print(f"  Generating {len(value_combinations)} run configurations.")

                    for combo in value_combinations:
                        current_run_config = single_value_params.copy()
                        current_run_config.update(dict(zip(param_names, combo)))
                        run_configs.append(current_run_config)
                
                # Add all configs for this method with their labels
                for i, config in enumerate(run_configs):
                    label = f"{method_name}"
                    if list_params:  # Add index if there are variations
                        label += f" (variation {i+1}/{len(run_configs)})"
                    
                    all_method_configs.append({
                        "method_name": method_name,
                        "config": config,
                        "label": label
                    })
                # --- End Parameter Variation Handling ---

            # Now, execute all methods for this seed (either sequentially or concurrently)
            if self.concurrent_execution and len(all_method_configs) > 1:
                print(f"\n--- Running {len(all_method_configs)} configurations concurrently (max {self.max_concurrent_methods} workers) ---")
                
                # Create a pool of worker threads
                with ThreadPoolExecutor(max_workers=self.max_concurrent_methods) as executor:
                    # Submit all tasks and keep track of futures
                    future_to_config = {}
                    for method_config in all_method_configs:
                        future = executor.submit(
                            self._run_method_with_config,
                            method_config["method_name"],
                            method_config["config"],
                            current_seed,
                            method_config["label"]
                        )
                        future_to_config[future] = method_config
                    
                    # Process results as they complete
                    for future in as_completed(future_to_config):
                        method_config = future_to_config[future]
                        try:
                            run_data = future.result()
                            results_list.append(run_data)
                        except Exception as e:
                            # Handle any exceptions that weren't caught in _run_method_with_config
                            error_msg = f"CRITICAL ERROR running {method_config['label']}: {e}"
                            print(error_msg)
                            print(traceback.format_exc())
                            
                            # Create a minimal error entry
                            error_data = {
                                "method": method_config["method_name"],
                                "issue": self.issue,
                                "statement": "CRITICAL ERROR",
                                "config_file": str(self.config_path),
                                "error_message": str(e),
                                "seed": current_seed,
                                **{f"param_{k}": v for k, v in method_config["config"].items()},
                            }
                            results_list.append(error_data)
            else:
                # Sequential execution (original logic)
                for method_config in all_method_configs:
                    method_name = method_config["method_name"]
                    config = method_config["config"]
                    label = method_config["label"]
                    
                    # Use the helper method to run this configuration
                    run_data = self._run_method_with_config(method_name, config, current_seed, label)
                    results_list.append(run_data)

        # --- Save results (no changes needed here) ---
        if not results_list:
            print("Warning: No results were generated.")
            return None

        results_df = pd.DataFrame(results_list)
        # Define column order for better readability
        core_cols = ["method", "statement", "error_message", "seed"]
        util_cols = sorted(
            [col for col in results_df.columns if col.startswith("utility_")]
        )
        param_cols = sorted(
            [col for col in results_df.columns if col.startswith("param_")]
        )
        other_cols = sorted(
            [
                col
                for col in results_df.columns
                if col
                not in core_cols + util_cols + param_cols + ["issue", "config_file"]
            ]
        )
        # Make sure 'generation_time_s' is included if present
        if (
            "generation_time_s" in results_df.columns
            and "generation_time_s" not in other_cols
        ):
            other_cols.append("generation_time_s")
            other_cols.sort()

        ordered_cols = (
            core_cols + util_cols + param_cols + ["issue", "config_file"] + other_cols
        )
        # Ensure all columns exist before reordering
        final_cols = [col for col in ordered_cols if col in results_df.columns]
        results_df = results_df[final_cols]

        results_path = self.run_output_dir / "results.csv"
        try:
            results_df.to_csv(results_path, index=False)
            print(f"\nSaved results summary to {results_path}")
        except Exception as e:
            print(f"Error saving results to {results_path}: {e}")
            # Fallback: Print results to console if saving fails
            print("\n--- Results (Failed to Save to CSV) ---")
            print(results_df.to_string())
            return results_df  # Still return the dataframe

        print("--- Experiment Run Finished ---")
        return results_df