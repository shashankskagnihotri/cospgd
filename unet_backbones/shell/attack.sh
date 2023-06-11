#!/bin/bash
#SBATCH -p gpu20,gpu22
#SBATCH -t 7:00:00 
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python attack.py \
    --attack $1 \
    --iterations $2 \
    --targeted $3 \
    --norm $4 \
    --alpha $5 \
    --epsilon $6 \
    --data_root /BS/asym_filter/nobackup/datasets_segmentation

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime