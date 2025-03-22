#!/bin/bash
#SBATCH -J train
#SBATCH -p local
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log

source ./.env/bin/activate
python ./tools/train.py