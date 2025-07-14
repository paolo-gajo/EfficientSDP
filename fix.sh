#!/bin/bash
#SBATCH -J gnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=./.slurm/%A/%a_output.log
#SBATCH --error=./.slurm/%A/%a_error.log
#SBATCH --mem=64g
#SBATCH --array=0-1
slurm_dir=./.slurm/$SLURM_ARRAY_JOB_ID
mkdir -p $slurm_dir
echo Creating directory: $slurm_dir
nvidia-smi
module load rust gcc arrow
. .env/bin/activate
declare -a cmd_list=(
# 'python3 ./src/train.py --opts --save_suffix transformer_0_113775/69 --seed 5 --use_gnn_steps 0 --gnn_layers 0 --parser_type simple --top_k 1 --arc_norm 1 --gnn_dropout 0 --gnn_activation tanh --dataset_name enewt --parser_rnn_layers 1 --parser_rnn_type transformer --training_steps 10000 --eval_steps 500 --use_tagger_rnn 1 --use_parser_rnn 1 --parser_rnn_hidden_size 400 --use_pred_tags 0'
# 'python3 ./src/train.py --opts --save_suffix transformer_0_113775/51 --seed 5 --use_gnn_steps 0 --gnn_layers 0 --parser_type simple --top_k 1 --arc_norm 0 --gnn_dropout 0 --gnn_activation tanh --dataset_name enewt --parser_rnn_layers 1 --parser_rnn_type transformer --training_steps 10000 --eval_steps 500 --use_tagger_rnn 1 --use_parser_rnn 1 --parser_rnn_hidden_size 400 --use_pred_tags 0'
# 'python3 ./src/train.py --opts --save_suffix transformer_0_113775/70 --seed 5 --use_gnn_steps 0 --gnn_layers 0 --parser_type simple --top_k 1 --arc_norm 1 --gnn_dropout 0 --gnn_activation tanh --dataset_name enewt --parser_rnn_layers 2 --parser_rnn_type transformer --training_steps 10000 --eval_steps 500 --use_tagger_rnn 1 --use_parser_rnn 1 --parser_rnn_hidden_size 400 --use_pred_tags 0'
# 'python3 ./src/train.py --opts --save_suffix transformer_0_113775/71 --seed 5 --use_gnn_steps 0 --gnn_layers 0 --parser_type simple --top_k 1 --arc_norm 1 --gnn_dropout 0 --gnn_activation tanh --dataset_name enewt --parser_rnn_layers 3 --parser_rnn_type transformer --training_steps 10000 --eval_steps 500 --use_tagger_rnn 1 --use_parser_rnn 1 --parser_rnn_hidden_size 400 --use_pred_tags 0'
# 'python3 ./src/train.py --opts --save_suffix transformer_0_113775/52 --seed 5 --use_gnn_steps 0 --gnn_layers 0 --parser_type simple --top_k 1 --arc_norm 0 --gnn_dropout 0 --gnn_activation tanh --dataset_name enewt --parser_rnn_layers 2 --parser_rnn_type transformer --training_steps 10000 --eval_steps 500 --use_tagger_rnn 1 --use_parser_rnn 1 --parser_rnn_hidden_size 400 --use_pred_tags 0'
# 'python3 ./src/train.py --opts --save_suffix transformer_0_113775/53 --seed 5 --use_gnn_steps 0 --gnn_layers 0 --parser_type simple --top_k 1 --arc_norm 0 --gnn_dropout 0 --gnn_activation tanh --dataset_name enewt --parser_rnn_layers 3 --parser_rnn_type transformer --training_steps 10000 --eval_steps 500 --use_tagger_rnn 1 --use_parser_rnn 1 --parser_rnn_hidden_size 400 --use_pred_tags 0'
)

echo ${cmd_list[${SLURM_ARRAY_TASK_ID}]}

${cmd_list[${SLURM_ARRAY_TASK_ID}]}