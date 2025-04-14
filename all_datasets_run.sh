#!/bin/bash
#SBATCH -J train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log
#SBATCH --account=def-hsajjad

source ./.env/bin/activate

training='steps'
training_steps=2000
eval_steps=100

freeze_encoder=0
learning_rate=1e-4

dataset_name_list=('ade' 'conll04' 'scierc' 'yamakata')

seed_list=(0 1 2 3)

for seed in "${seed_list[@]}"; do
    for dataset_name in "${dataset_name_list[@]}"; do
        python ./tools/train.py --opts \
        --training $training \
        --training_steps $training_steps \
        --eval_steps $eval_steps \
        --freeze_encoder $freeze_encoder \
        --learning_rate $learning_rate \
        --dataset_name $dataset_name \
        --seed $seed
    done
done