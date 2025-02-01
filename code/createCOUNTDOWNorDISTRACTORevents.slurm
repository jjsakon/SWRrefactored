#!/bin/bash
#SBATCH --output=/home1/john/SWRrefactored/cluster/log-%j.out
#SBATCH --error=/home1/john/SWRrefactored/cluster/log-%j.err
#SBATCH --nodes=1
#SBATCH --mem=49GB
#SBATCH --array=0-616%10 # 0-700%6 # FR1 567; catFR1 615

echo "Running task number $SLURM_ARRAY_TASK_ID"
python -u /home1/john/SWRrefactored/code/createCOUNTDOWNorDISTRACTORevents.py $SLURM_ARRAY_TASK_ID
