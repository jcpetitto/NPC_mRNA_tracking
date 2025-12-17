#!/bin/bash

#BSUB -q gpu
#BSUB -n 4
#BSUB -W 04:00
#BSUB -M 32000
#BSUB -o "logs/output_%J_%I.log"
#BSUB -e "logs/error_%J_%I.log"


# Create the log directory in the submission directory
# Note: This will be relative to where you ran the 'bsub' command
mkdir -p logs

echo "Starting job $LSB_JOBINDEX on host $HOSTNAME"
echo "Date: $(date)"

# --- CONDA SETUP ---
# 1. Initialize Conda using the passed environment variable (FIX 1)
source /pi/david.grunwald-umw/miniconda3/etc/profile.d/conda.sh

# 2. Activate your specific Conda environment
# CONDA_ENV_NAME="pyr_yeast_env"
conda activate pyr_yeast_env

# 3. Define the explicit path to the Python executable for robustness (FIX 2)
# PYTHON_EXEC="$CONDA_BASE_PATH/envs/$CONDA_ENV_NAME/bin/python"

# echo "Using conda environment: $CONDA_DEFAULT_ENV"
# echo "Using Python executable: $PYTHON_EXEC"

# --- CONSTRUCT ABSOLUTE PATH TO THE MAIN PYTHON SCRIPT ---
# PYTHON_SCRIPT="$SCRIPT_DIR/../main_cluster_dual_label.py"
PYTHON_SCRIPT="/home/jocelyn.tourtellotte-umw/src_yeast_pipeline/src_python/main_cluster.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Main python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# --- Run your Python script using the explicit, absolute paths ---
python "$PYTHON_SCRIPT" $LSB_JOBINDEX "$CONFIG_FILE"

# --- Deactivate ---
conda deactivate

echo "Job $LSB_JOBINDEX finished with exit code $?"