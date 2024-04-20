#!/bin/bash
#SBATCH --partition=SCSEGPU_M2
#SBATCH --qos=q_dmsai
#SBATCH --nodes=1
#SBATCH	--cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --job-name=nlp_2
#SBATCH --output=outputs/%x.out
#SBATCH --error=outputs/%x.err
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate aug
wandb login ed3d1ea0f2dcae60313913eb8ec2893502a12246
python -m training 
	
