import yaml
import argparse
from pathlib import Path
import sys
import pandas as pd
import logging

# Add the src directory to the Python path
# This allows importing modules from src like 'from src.experiment import Experiment'
# Adjust the path if your script is not directly in paper_experiments/
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    print(f"Added {src_path} to sys.path")


# Now we can import from src
try:
    from src.experiment import Experiment

    print("Successfully imported Experiment.")
except ImportError as e:
    print(f"Error importing Experiment class: {e}")
    print(
        "Ensure the src directory is in your Python path and contains __init__.py and experiment.py"
    )
    sys.exit(1)


def run_experiment_from_config(config_path_str: str):
    """Loads config and runs the experiment."""
    config_path = Path(config_path_str)
    print(f"Attempting to load configuration from: {config_path}")

    if not config_path.is_file():
        print(f"ERROR: Configuration file not found at {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if config is None:  # Handle empty config file
            print(f"ERROR: Configuration file {config_path} is empty or invalid.")
            sys.exit(1)
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse YAML configuration file {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load configuration file {config_path}: {e}")
        sys.exit(1)

    # --- Central Logging Configuration ---
    # Configure the root logger ONCE. Set its level low (DEBUG)
    # so that it doesn't filter messages prematurely.
    # Specific module loggers will control their own verbosity via setLevel.
    logging.basicConfig(
        level=logging.DEBUG,  # <<< Root level can remain DEBUG for your app logs
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        # Optional: Add filename='experiment.log' to log to a file
        # Optional: force=True can be useful if something else might have already called basicConfig
        # force=True
    )
    # Log a message indicating the root setup (optional)
    logging.info("Root logger configured with handler level DEBUG.")

    # --- Suppress ALL logs from the 'together' library ---
    # Get the logger used by the 'together' library
    together_logger = logging.getLogger("together")
    # Set its level to CRITICAL to ignore DEBUG, INFO, WARNING, and ERROR messages
    together_logger.setLevel(logging.CRITICAL)
    logging.info(
        f"Set 'together' library logger level to {logging.getLevelName(together_logger.level)} to suppress its logs."
    )

    # --- Suppress DEBUG/INFO logs from 'urllib3' library ---
    urllib3_logger = logging.getLogger("urllib3")
    urllib3_logger.setLevel(logging.WARNING)  # Ignore DEBUG and INFO messages
    logging.info(
        f"Set 'urllib3' library logger level to {logging.getLevelName(urllib3_logger.level)}"
    )
    # --- End Library Log Suppression ---

    # --- Instantiate and Run Experiment ---
    try:
        experiment = Experiment(config)
        results_df = experiment.run()  # run() now handles printing finish message

        if results_df is not None:
            print("\n--- Experiment Results Summary ---")
            # Display more rows/cols if needed
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                1000,
            ):
                print(results_df)
        else:
            print(
                "Experiment finished, but no results DataFrame was returned (check logs for errors)."
            )

    except Exception as e:
        print(f"\n--- CRITICAL ERROR DURING EXPERIMENT EXECUTION ---")
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run paper experiments based on a configuration file."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/experiment_config.yaml",
        help="Path to the experiment configuration YAML file (default: configs/experiment_config.yaml)",
    )
    args = parser.parse_args()

    # Run the experiment using the configuration file
    run_experiment_from_config(args.config)
