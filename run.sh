#!/bin/bash
#SBATCH -J train
#SBATCH -p local
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log

augment_train=1
augment_val=0
augment_test=0

augment_k_train=(
    1
    5
    10
    20
    40
    60
    80
    100
    )
augment_k_val=0
augment_k_test=0

keep_og_train=1
keep_og_val=1
keep_og_test=1

use_bert_positional_embeddings=1
use_tag_embeddings_in_parser=1
use_tagger_lstm=0
use_parser_lstm=0
use_gnn=0
use_step_mask=0

freeze_encoder=0
learning_rate=1e-4

for k in "${augment_k_train[@]}"; do
    echo "Training with augment_k_train = $k"
    python ./tools/train.py --opts \
    --augment_train $augment_train \
    --augment_val $augment_val \
    --augment_test $augment_test \
    --augment_k_train $k \
    --augment_k_val $augment_k_val \
    --augment_k_test $augment_k_test \
    --keep_og_train $keep_og_train \
    --keep_og_val $keep_og_val \
    --keep_og_test $keep_og_test \
    --use_bert_positional_embeddings $use_bert_positional_embeddings \
    --use_tag_embeddings_in_parser $use_tag_embeddings_in_parser \
    --use_tagger_lstm $use_tagger_lstm \
    --use_parser_lstm $use_parser_lstm \
    --use_gnn $use_gnn \
    --use_step_mask $use_step_mask \
    --freeze_encoder $freeze_encoder \
    --learning_rate $learning_rate
    
done