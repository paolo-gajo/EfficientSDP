#!/bin/bash
#SBATCH -J gnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=./.slurm/%A/%a_output.log
#SBATCH --error=./.slurm/%A/%a_error.log
#SBATCH --mem=64g
#SBATCH --array=0-N

slurm_dir="./.slurm/$SLURM_ARRAY_JOB_ID"
mkdir -p $slurm_dir
echo "Creating directory: $slurm_dir"
nvidia-smi
module load rust gcc arrow
. .env/bin/activate

# Cartesian product function
cartesian_product() {
    local result=("")
    local -n arrays=$1
    
    for array_name in "${arrays[@]}"; do
        local -n current_array=$array_name
        local new_result=()
        
        for existing in "${result[@]}"; do
            for item in "${current_array[@]}"; do
                new_result+=("${existing:+$existing,}$item")
            done
        done
        result=("${new_result[@]}")
    done
    
    printf '%s\n' "${result[@]}"
}
declare -a seed=(
    0
    1
    2
    # 3
    # 4
)
# Define parameter arrays
declare -a use_gnn_steps_opts=(0)
declare -a rnn_layers_opts=(
    0
    1
    2
    3
    )
declare -a gnn_layers_opts=(
    0
    1
    2
    3
    )
declare -a parser_type_opts=(
    gat
    # simple
    )
declare -a parser_rnn_type_opts=(
    # gru
    lstm
    # rnn
    # normlstm
    # normrnn
    # transformer
)
declare -a top_k_opts=(
    # 1
    # 2
    # 3
    4
    )
declare -a arc_norm_opts=(
    0
    1
    )
declare -a gnn_dropout_opts=(
    0
    # 0.3
    )
declare -a gnn_activation_opts=(tanh)
declare -a dataset_name_opts=(
    ade
    conll04
    scierc
    erfgc
    scidtb
    enewt
  )

declare -a rnn_residual=(
    0
    # 1
    )

# Generate all combinations
array_names=(
            seed
            use_gnn_steps_opts
            gnn_layers_opts
            parser_type_opts
            top_k_opts
            arc_norm_opts
            gnn_dropout_opts
            gnn_activation_opts
            dataset_name_opts
            rnn_layers_opts
            parser_rnn_type_opts
            rnn_residual
            )
combinations=$(cartesian_product array_names)

{
    for array_name in "${array_names[@]}"; do
        # Access array by name using indirect expansion
        values="${array_name}[@]"
        echo "$array_name: ${!values}"
    done
} > "${slurm_dir}/hyperparameters.txt"

# Training parameters
training_steps=5000
eval_steps=500

save_suffix=gnn_lstm

use_tagger_rnn=1
use_parser_rnn=1

use_pred_tags=1

# Convert combinations to commands
declare -a commands=()
while IFS= read -r combo; do
    IFS=',' read -ra params <<< "$combo"

    if [[ "${params[8]}" == "scidtb" || "${params[8]}" == "enewt" ]]; then
        use_pred_tags=0
    else
        use_pred_tags=1
    fi

    # if [[ "${params[9]}" == 0 && "${params[10]}" != 'lstm' ]]; then
    #     continue
    # fi
    
    cmd="python ./src/train.py
                --opts
                --save_suffix ${save_suffix}_${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
                --seed ${params[0]}
                --use_gnn_steps ${params[1]}
                --gnn_layers ${params[2]}
                --parser_type ${params[3]}
                --top_k ${params[4]}
                --arc_norm ${params[5]}
                --gnn_dropout ${params[6]}
                --gnn_activation ${params[7]}
                --dataset_name ${params[8]}
                --parser_rnn_layers ${params[9]}
                --parser_rnn_type ${params[10]}
                --rnn_residual ${params[11]}
                --training_steps $training_steps 
                --eval_steps $eval_steps
                --use_tagger_rnn $use_tagger_rnn
                --use_parser_rnn $use_parser_rnn
                --parser_rnn_hidden_size 400
                --use_pred_tags $use_pred_tags
                "
    if [[ "${params[1]}" -gt 0  && "${params[3]}" == 'simple' ]]; then
        continue
    fi
    # if [[ "${params[2]}" == 0 && "${params[4]}" -gt 1 ]]; then
    #     echo here
    #     continue
    # fi
    # echo ${cmd}
    commands+=("$cmd")
done <<< "$combinations"

total_combinations=${#commands[@]}

if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
    command_to_run="${commands[$SLURM_ARRAY_TASK_ID]}"
    # echo "$command_to_run"
    $command_to_run
elif [[ $1 ]]; then
    for (( i=start; i<${#commands[@]}; i++ ))
    do
        echo "$((i+1)) of ${#commands[@]}"
        cmd="${commands[$i]}"
        echo "${cmd}"
        $cmd
    done
else
    echo "This script should be run as a SLURM array job."
    echo "Use: sbatch --array=0-$((total_combinations-1)) $0"
    echo "This will distribute $total_combinations jobs across N GPUs."
fi
