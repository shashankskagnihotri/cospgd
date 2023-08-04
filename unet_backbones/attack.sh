#!/bin/bash

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python attack.py --attack $1 --iterations $2 --targeted $3 --norm $4 --alpha $5 --epsilon $6

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime