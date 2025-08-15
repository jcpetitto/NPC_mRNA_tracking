#!/bin/bash
echo "Counting experiments to determine job array size..."

# --- Calculate Number of Jobs ---
# Count the JSON files directly.
# This is faster and avoids Python path issues.
NUM_JOBS=$(ls -1 /home/jocelyn.tourtellotte-umw/yeast_output/tracking_experiments/*.json | wc -l)

# make sure this file is executable
chmod +x ./src_yeast_pipeline/shell_scripts/run_pipeline.sh
# --- Submit the Job Array ---
# Check if we found any jobs to run
if [ "$NUM_JOBS" -gt 0 ]; then
    echo "Found $NUM_JOBS experiments. Submitting job array..."
    
    # Submit the worker script with the job name and calculated array size
    bsub -J "fov_processing[1-$NUM_JOBS]" < ./src_yeast_pipeline/shell_scripts/run_pipeline.sh
    
    echo "Job array submitted."
else
    echo "Error: Could not find any experiments to process. No jobs submitted."
fi