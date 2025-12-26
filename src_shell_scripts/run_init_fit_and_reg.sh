#!/bin/bash

# --- LSF Directives: Requesting Resources ---

# Set the job name
#BSUB -J YeastAnalysis

# Specify the output and error files. %J is the job ID.
#BSUB -o yeast_analysis_%J.out
#BSUB -e yeast_analysis_%J.err

# Request a queue that has GPU resources. Common names are 'gpu', 'gpu_queue'.
# You may need to change this depending on your cluster's configuration.
#BSUB -q gpu

# Request one GPU in exclusive process mode.
# This ensures the GPU is dedicated to your job.
#BSUB -gpu "num=1:mode=exclusive_process"

# Request the number of CPU cores
#BSUB -n 4

# Request memory for the job
#BSUB -R "rusage[mem=32GB]"

# Set a wall-clock time limit for the job
#BSUB -W 8:00

# --- Job Execution ---

# Print job information for logging
echo "------------------------------------------------"
echo "Job started on: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo "------------------------------------------------"

# Load the Conda module (this is often required on HPC clusters)
# The exact module name might differ slightly.
module load conda

# Activate your Conda environment
conda activate yeast_env_new

# Navigate to the directory containing your script, if necessary
# cd /home/jocelyn.tourtellotte-umw/src_yeast_pipeline/

# Run your Python script
# Make sure cluster_main.py is in the directory where you run bsub,
# or provide the full path to it.
python /home/jocelyn.tourtellotte-umw/src_yeast_pipeline/main_cluster.py

echo "------------------------------------------------"
echo "Job finished on: $(date)"
echo "------------------------------------------------"
