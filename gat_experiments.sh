#!/bin/bash
#SBATCH -J gnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=./.slurm/%A/%a_output.log
#SBATCH --error=./.slurm/%A/%a_error.log
#SBATCH --mem=64g
#SBATCH --array=0-N%999

mkdir -p "./.slurm/$SLURM_ARRAY_JOB_ID"
echo "Creating directory: ./.slurm/$SLURM_ARRAY_JOB_ID"
nvidia-smi
module load rust gcc arrow
. .env/bin/activate

declare -a use_gnn_steps_opts=(
    1000
    0
    )
declare -a gnn_enc_layers_opts=(
    0
    1
    2
    3
    )
declare -a parser_type_opts=(
    # simple
    # gat_unbatched
    gat
    )

top_k=1
training_steps=20000
eval_steps=500
results_suffix=_gat

declare -a commands=()
for use_gnn_steps in "${use_gnn_steps_opts[@]}"
do
    for gnn_enc_layers in "${gnn_enc_layers_opts[@]}"
    do
        for parser_type in "${parser_type_opts[@]}"
        do
            cmd="python ./src/train.py
                        --opts --parser_type $parser_type
                        --training_steps $training_steps 
                        --use_gnn_steps $use_gnn_steps
                        --gnn_enc_layers $gnn_enc_layers
                        --results_suffix $results_suffix
                        --eval_steps $eval_steps
                        --use_tagger_rnn 0
                        --use_parser_rnn 0
                        --parser_rnn_hidden_size 400
                        --parser_rnn_layers 0
                        --top_k $top_k"
            commands+=("$cmd")
        done
    done
done

total_combinations=${#commands[@]}

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    command_to_run="${commands[$SLURM_ARRAY_TASK_ID]}"
    echo "$command_to_run"
    $command_to_run
else
  # If run manually, print the total number of combinations
  echo "This script should be run as a SLURM array job."
  echo "Use: sbatch --array=0-$((total_combinations-1))%999 $0"
  echo "This will distribute $total_combinations jobs across N GPUs."
fi