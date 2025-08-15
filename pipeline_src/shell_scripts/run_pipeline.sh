#!/bin/bash

#!/bin/bash

#BSUB -q gpu
#BSUB -n 4
#BSUB -W 04:00
#BSUB -M 32000
#BSUB -o "logs/output_%J_%I.log" # Note: %J is the job ID, %I is the array index
#BSUB -e "logs/error_%J_%I.log"

# Create the log directory if it doesn't exist
mkdir -p logs

echo "Starting job $LSB_JOBINDEX on host $HOSTNAME"
echo "Date: $(date)"

# --- CONDA SETUP ---
# 1. Initialize Conda for non-interactive shells
source /pi/david.grunwald-umw/miniconda3/etc/profile.d/conda.sh

# 2. Activate your specific Conda environment
conda activate yeast_env_new

echo "Using conda environment: $CONDA_DEFAULT_ENV"
# 3. Run your Python script
# The LSF job array index is passed as a command-line argument.
python ./src_yeast_pipeline/main_cluster.py $LSB_JOBINDEX

# 4. Deactivate the environment (good practice)
conda deactivate

echo "Job $LSB_JOBINDEX finished with exit code $?"