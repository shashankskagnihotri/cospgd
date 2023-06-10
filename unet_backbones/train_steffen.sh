#!/bin/bash
#SBATCH -p 
#SBATCH -t 11:59:59 
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --cpus_per_task=16
#SBATCH --array=0-2%3
#SBATCH --output=slurm/training/%A_%a.out

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID and array id $SLURM_ARRAY_TASK_ID on queue $SLURM_JOB_PARTITION";
start=`date +%s`

#echo "Training convnext_tiny on cityscapes Epochs 100 Small decoder True"
#python -W ignore train.py --epochs 100 --dataset cityscapes --small_decoder "True"

if [[ $SLURM_ARRAY_TASK_ID = "1" ]]
then
    #echo "Training convnext_tiny on cityscapes Epochs 100 Small decoder False"
    python -W ignore train.py --epochs 100 --dataset cityscapes --small_decoder "False"

elif [[ $SLURM_ARRAY_TASK_ID = "2" ]]
then
    #echo "Training convnext_tiny on cityscapes Epochs 150  Small decoder True"
    python -W ignore train.py --epochs 150 --dataset cityscapes --small_decoder "True"
elif [[ $SLURM_ARRAY_TASK_ID = "3" ]]
then
    #echo "Training convnext_tiny on cityscapes Epochs 150  Small decoder False"
    python -W ignore train.py --epochs 150 --dataset cityscapes --small_decoder "False"
fi

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime