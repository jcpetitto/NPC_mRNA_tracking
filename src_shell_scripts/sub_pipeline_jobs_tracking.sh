#!/bin/bash
echo "Initializing job submission..."
# --- 1. Define All Paths and Variables at the Top ---
# CONDA_BASE_PATH="/pi/david.grunwald-umw/miniconda3"
CONFIG_FILE="$1"

# Get the directory where THIS script is located
SCRIPT_DIR=/home/jocelyn.tourtellotte-umw/src_yeast_pipeline
SHELL_DIR="$SCRIPT_DIR/src_shell_scripts"

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

# echo "Using Conda installation at: $CONDA_BASE_PATH"
echo "Using config file: $CONFIG_FILE"

# --- 3. Activate Environment and Discover Jobs ---
echo "Activating Conda environment to discover jobs..."
# source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
source /pi/david.grunwald-umw/miniconda3/etc/profile.d/conda.sh
conda activate pyr_yeast_env

PYTHON_SCRIPT="$SCRIPT_DIR/src_python/gather_experiments.py"

NUM_JOBS=$(python "$PYTHON_SCRIPT" "$CONFIG_FILE" | tail -n 1)

conda deactivate

echo "Discovery complete."

# --- 4. Check Discovery Result ---
if ! [[ "$NUM_JOBS" =~ ^[0-9]+$ ]]; then
    echo "Error: gather_experiments.py did not return a valid number. Aborting."
    echo "--- Captured Output ---"
    echo "$NUM_JOBS"
    echo "-----------------------"
    exit 1
fi

# --- 5. Submit the Job Array ---
if [ "$NUM_JOBS" -gt 0 ]; then
    echo "Found $NUM_JOBS experiments. Submitting job array..."
    
        WORKER_SCRIPT="$SHELL_DIR/run_pipeline_tracking.sh"
    chmod +x "$WORKER_SCRIPT"
    
    echo "DEBUG: Submitting job directly (without '<' redirection)."
    # Now this line works because CONDA_BASE_PATH is defined
    # bsub -J "fov_proc[1-$NUM_JOBS]" -env "CONFIG_FILE=$CONFIG_FILE, CONDA_BASE_PATH=$CONDA_BASE_PATH" "$WORKER_SCRIPT"
    bsub -J "fov_proc[1-$NUM_JOBS]" -env "CONFIG_FILE=$CONFIG_FILE" "$WORKER_SCRIPT"
    
    echo "Job array submitted."
else
    echo "No experiments found to process. No jobs submitted."
fi