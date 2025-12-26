import sys
from imaging_pipeline import ImagingPipeline

# This script reads the pipeline config, counts the experiments,
# and prints the number to the console.

if __name__ == "__main__":
    # --- 1. Check for the command-line argument ---
    # sys.argv[0] is the script name, sys.argv[1] is the first argument.
    if len(sys.argv) < 2:
        # If no argument is provided, print an error and exit.
        print("Error: Please provide the path to the configuration file.", file=sys.stderr)
        sys.exit(1)

    # --- 2. Use the argument as the config path ---
    config_path = sys.argv[1]

    try:
        # Initialize the pipeline with the provided path
        pipeline = ImagingPipeline(config_path)
        all_experiments = pipeline.get_experiments()
        
        # Print the total number of experiments found
        num_experiments = len(all_experiments)
        print(num_experiments)
        
    except Exception as e:
        # Print errors to stderr and exit with an error code
        print(f"Error discovering jobs: {e}", file=sys.stderr)
        sys.exit(1)