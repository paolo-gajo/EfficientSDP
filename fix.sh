#!/bin/bash
#SBATCH -J gnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
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
'python3 ./src/train.py --opts --results_suffix 62454355/68 --seed 2 --use_gnn_steps 0 --gnn_layers 0 --parser_type gat --top_k 4 --arc_norm 0 --gnn_dropout 0 --gnn_activation tanh --dataset scidtb --training_steps 10000 --eval_steps 500 --use_tagger_rnn 0 --use_parser_rnn 0 --parser_rnn_hidden_size 400 --parser_rnn_layers 0 --use_pred_tags 0'
)

echo ${cmd_list[${SLURM_ARRAY_TASK_ID}]}

${cmd_list[${SLURM_ARRAY_TASK_ID}]}