#!/bin/bash
echo "Initializing job submission..."
# vars from CL args
CONFIG_FILE="$1"
RERUN_FLAG="$2"

# directory where THIS script is located
SCRIPT_DIR="your/home/src_yeast_pipeline"
SHELL_DIR="$SCRIPT_DIR/src_shell_scripts"

PATH_TO_CONDA="path/to/miniconda3/etc/profile.d/conda.sh"

# --- 2. Check Command-Line Argument ---
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: You must provide a path to a configuration file."
    echo "Usage: $0 /path/to/config.json"
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at: $CONFIG_FILE"
    exit 1
fi

# checking paths for debugging job submission if necessary
# echo "Using Conda installation at: $CONDA_BASE_PATH"
# echo "Using config file: $CONFIG_FILE"

# activate conda environment so the number of jobs can be
# determined by a python script based on the config file
echo "Activating Conda environment to discover jobs..."
source $PATH_TO_CONDA
conda activate pyr_yeast_env

PYTHON_SCRIPT="$SCRIPT_DIR/src_python/gather_experiments.py"

NUM_JOBS=$(python "$PYTHON_SCRIPT" "$CONFIG_FILE" | tail -n 1)

conda deactivate

echo "Discovery complete."

# checks the number of experiments found (becomes # of jobs in array)
if ! [[ "$NUM_JOBS" =~ ^[0-9]+$ ]]; then
    echo "Error: gather_experiments.py did not return a valid number. Aborting."
    echo "--- Captured Output ---"
    echo "$NUM_JOBS"
    echo "-----------------------"
    exit 1
fi

# submit job array
if [ "$NUM_JOBS" -gt 0 ]; then
    echo "Found $NUM_JOBS experiments. Submitting job array..."
    
    WORKER_SCRIPT="$SHELL_DIR/run_pipeline.sh"
    chmod +x "$WORKER_SCRIPT"
    
    echo "DEBUG: Submitting job directly (without '<' redirection)."

    # set the configuration file path as an environmental variable
    #   that the worker script can access
    bsub -J "fov_proc[1-$NUM_JOBS]" -env "CONFIG_FILE=$CONFIG_FILE,RERUN_FLAG=$RERUN_FLAG" "$WORKER_SCRIPT"
    
    echo "Job array submitted."
else
    echo "No experiments found to process. No jobs submitted."
fi