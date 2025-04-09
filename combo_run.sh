#!/bin/bash
#SBATCH -J train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log
#SBATCH --mem=64G

# source ./.env/bin/activate

augment_train=0
augment_val=0
augment_test=0

augment_k_train=(
    0
    # 1
    # 5
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
use_tag_embeddings_in_parser=0
use_tagger_lstm=0
use_parser_lstm=0
use_gnn=0
use_step_mask=0
laplacian_pe=0

augment_type_toggle=('random'
                    'permute'
                    'hybrid')
training='steps'
training_steps=2000
eval_steps=100
freeze_encoder=1
learning_rate=1e-3

seed_list=(
    0
    1
    2
    3
    4
    )

parser_type_toggle=(
    'mtrfg'
    'gnn'
    )
    
gnn_enc_layers=1

for seed in "${seed_list[@]}"; do
    for parser_type in "${parser_type_toggle[@]}"; do
    python ./tools/train.py --opts \
        --use_tag_embeddings_in_parser $use_tag_embeddings_in_parser \
        --use_tagger_lstm $use_tagger_lstm \
        --use_parser_lstm $use_parser_lstm \
        --training $training \
        --training_steps $training_steps \
        --eval_steps $eval_steps \
        --freeze_encoder $freeze_encoder \
        --learning_rate $learning_rate \
        --seed $seed \
        --parser_type $parser_type \
        --gnn_enc_layers $gnn_enc_layers
    done
done