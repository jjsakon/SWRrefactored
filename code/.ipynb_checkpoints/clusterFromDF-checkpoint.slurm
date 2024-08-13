#!/bin/bash
#SBATCH --output=/home1/john/SWRrefactored/cluster/log-%j.out
#SBATCH --error=/home1/john/SWRrefactored/cluster/log-%j.err
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --array=0-30%6

echo "Running task number $SLURM_ARRAY_TASK_ID"
python -u /home1/john/SWRrefactored/code/createEventsForDF.py $SLURM_ARRAY_TASK_ID
