#!/bin/bash
#SBATCH -J train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log
#SBATCH --mem=64G
source ./.env/bin/activate

augment_train=1
augment_val=0
augment_test=0

augment_k_train=(1
    1
    5
    # 10
    # 20
    # 40
    # 60
    # 80
    # 100
    )

augment_k_val=0
augment_k_test=0

keep_og_train=1
keep_og_val=1
keep_og_test=1

use_bert_positional_embeddings=1
use_abs_step_embeddings=0
use_tag_embeddings_in_parser=1
use_tagger_lstm=0
use_parser_lstm=0
use_gnn=0
use_step_mask=0
laplacian_pe=0

freeze_encoder=1
learning_rate=1e-3

augment_type_toggle=('random'
                    'permute'
                    'hybrid')
training='steps'
training_steps=2000
eval_steps=100

seed_list=(0 1 2 3 4)

procedural=1

base_params="--training $training \
        --training_steps $training_steps \
        --eval_steps $eval_steps \
        --freeze_encoder $freeze_encoder \
        --learning_rate $learning_rate \
        --seed $seed \
        --procedural $procedural"

for seed in "${seed_list[@]}"; do
    python ./tools/train.py --opts $base_params
done

for seed in "${seed_list[@]}"; do
    for k in "${augment_k_train[@]}"; do
        python ./tools/train.py --opts \
        $base_params \
        --augment_train $augment_train \
        --augment_val $augment_val \
        --augment_test $augment_test \
        --augment_k_train $k \
        --augment_k_val $augment_k_val \
        --augment_k_test $augment_k_test \
        --keep_og_train $keep_og_train \
        --keep_og_val $keep_og_val \
        --keep_og_test $keep_og_test \
        --augment_type $1
    done
done