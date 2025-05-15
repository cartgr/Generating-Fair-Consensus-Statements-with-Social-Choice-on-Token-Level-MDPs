#!/usr/bin/env python3
import argparse
import os
import sys
import glob
import time
from pathlib import Path
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("neurips_experiments.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def find_config_files(base_dir, model=None, scenario=None, method=None):
    """Find all config files matching the specified filters."""
    # Start with the base directory
    config_path = Path(base_dir)

    # Apply model filter if specified
    if model:
        config_path = config_path / model
    else:
        model_pattern = "*"

    # Apply scenario filter if specified
    if scenario:
        if isinstance(scenario, int):
            scenario = f"scenario_{scenario}"
        config_path = config_path / scenario
    else:
        scenario_pattern = "scenario_*"

    # Apply method filter if specified
    if method:
        method_pattern = f"{method}.yaml"
    else:
        method_pattern = "*.yaml"

    # Construct the glob pattern
    if model and scenario:
        pattern = str(config_path / method_pattern)
    elif model:
        pattern = str(config_path / scenario_pattern / method_pattern)
    else:
        pattern = str(config_path / model_pattern / scenario_pattern / method_pattern)

    # Find matching files
    config_files = sorted(glob.glob(pattern))
    return config_files


def run_experiment(config_file):
    """Run a single experiment with evaluation."""
    logger.info(f"Starting experiment with config: {config_file}")
    start_time = time.time()

    try:
        # Run the experiment with evaluation
        cmd = ["python", "run_experiment_with_eval.py", "--config", config_file]

        # Execute the command and pass through output directly
        process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,  # Pass through to console
            stderr=sys.stderr,  # Pass through to console
        )

        process.wait()

        if process.returncode != 0:
            logger.error(f"Experiment failed with return code {process.returncode}")
            return False

        duration = time.time() - start_time
        logger.info(f"Experiment completed in {duration:.2f} seconds")

        return True
    except Exception as e:
        logger.error(f"Unexpected error running experiment: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all NeurIPS experiments")
    parser.add_argument(
        "--model",
        type=str,
        choices=["gemma", "llama"],
        help="Only run experiments for a specific model",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Only run experiments for a specific scenario",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "habermas_vs_best_of_n",
            "habermas_only",
            "beam_search",
            "finite_lookahead",
        ],
        help="Only run experiments for a specific method",
    )

    args = parser.parse_args()

    # Find the config files to run
    config_files = find_config_files(
        "configs/appendix",
        model=args.model,
        scenario=f"scenario_{args.scenario}" if args.scenario else None,
        method=args.method,
    )

    if not config_files:
        logger.error("No matching config files found!")
        return

    logger.info(f"Found {len(config_files)} config files to process")

    for idx, config_file in enumerate(config_files, 1):
        logger.info(f"Running experiment {idx}/{len(config_files)}: {config_file}")
        print("\n" + "=" * 80)
        print(f"EXPERIMENT {idx}/{len(config_files)}: {config_file}")
        print("=" * 80 + "\n")

        success = run_experiment(config_file)

        if not success:
            logger.warning(f"Failed to run experiment: {config_file}")

        # Add some space after each experiment
        print("\n" + "-" * 80 + "\n")

    logger.info("All experiments completed!")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    logger.info(f"Starting NeurIPS experiments at {datetime.now()}")
    main()
    logger.info(f"Finished NeurIPS experiments at {datetime.now()}")
    print("=" * 80 + "\n")
