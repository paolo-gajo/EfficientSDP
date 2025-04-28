#!/bin/bash
#SBATCH -J train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log

# parser_opts=('mtrfg' 'gnn')

source .env/bin/activate

python ./tools/train.py # --opt --use_tag_embeddings_in_parser 0 --use_tagger_rnn 0 --use_parser_rnn 1 --training steps --training_steps 2000 --eval_steps 100 --freeze_encoder 1 --seed 1 --parser_type simple --gnn_enc_layers 0 --arc_norm 1 --parser_rnn_type lstm --model_name bert-base-uncased --use_lora 0 --parser_residual 0 --parser_rnn_layers 2 --parser_rnn_hidden_size 200 --dataset_name yamakata --results_suffix _lstm_size_ablations --arc_representation_dim 300