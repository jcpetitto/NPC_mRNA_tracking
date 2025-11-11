#!/bin/bash

#BSUB -q gpu
#BSUB -n 4
#BSUB -W 24:00
#BSUB -M 32000
#BSUB -o "/home/jocelyn.tourtellotte-umw/yeast_output/logs/output_%J_%I.log"
#BSUB -e "/home/jocelyn.tourtellotte-umw/yeast_output/logs/error_%J_%I.log"

# Create the log directory
mkdir -p "/home/jocelyn.tourtellotte-umw/yeast_output/logs"

HOME_DIR="/home/jocelyn.tourtellotte-umw"
DATA_DIR="/pi/david.grunwald-umw"
# SIF_PATH="${HOME_DIR}/yeast.sif"
SIF_PATH="/home/jocelyn.tourtellotte-umw/yeastainer.sif"
# PYTHON_SCRIPT="${HOME_DIR}/src_yeast_pipeline/src_python/main_cluster_dual_label.py"
PYTHON_SCRIPT="/home/jocelyn.tourtellotte-umw/src_yeast_pipeline/src_python/main_cluster_dual_label.py"

echo "Starting job $LSB_JOBINDEX on host $HOSTNAME"
echo "Date: $(date)"

singularity exec --nv \
    -B ${HOME_DIR}:${HOME_DIR} \
    -B ${DATA_DIR}:${DATA_DIR} \
    ${SIF_PATH} \
    python -u ${PYTHON_SCRIPT} ${LSB_JOBINDEX} ${CONFIG_FILE}
    # use u flag to unbuffer python output

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Main python script not found at $PYTHON_SCRIPT"
    exit 1
fi

echo "Job $LSB_JOBINDEX finished with exit code $?"