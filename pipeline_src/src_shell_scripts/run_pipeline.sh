#!/bin/bash

#BSUB -q gpu
#BSUB -n 4
#BSUB -W 48:00
#BSUB -M 32000
#BSUB -o "your/home/logs/output_%J_%I.log"
#BSUB -e "your/home/logs/error_%J_%I.log"

# Create the log directory
mkdir -p "your/home/logs"

HOME_DIR="your/home/"
DATA_DIR="data/storage/"
SIF_PATH="your/home/yeastainer.sif"
SCRIPT_DIR="your/home/src_yeast_pipeline"
PYTHON_SCRIPT="$SCRIPT_DIR/src_python/main_cluster.py"

echo "Starting job $LSB_JOBINDEX on host $HOSTNAME"
echo "Date: $(date)"

singularity exec --nv \
    -B ${HOME_DIR}:${HOME_DIR} \
    -B ${DATA_DIR}:${DATA_DIR} \
    ${SIF_PATH} \
    python -u ${PYTHON_SCRIPT} ${LSB_JOBINDEX} ${CONFIG_FILE} ${RERUN_FLAG}
    # use u flag to unbuffer python output

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Main python script not found at $PYTHON_SCRIPT"
    exit 1
fi

echo "Job $LSB_JOBINDEX finished with exit code $?"