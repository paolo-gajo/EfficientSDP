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
    # 0
    1
    2
    3
    4
)
# Define parameter arrays
declare -a use_gnn_steps_opts=(0)
declare -a gnn_enc_layers_opts=(
    0
    1
    2
    3
    )
declare -a parser_type_opts=(
    gat
    # simple
    )
declare -a top_k_opts=(
    1
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
    0.3
    )
declare -a gnn_activation_opts=(tanh)
declare -a dataset_name_opts=(
#   "ade"
#   "conll04"
#   "scierc"
#   "erfgc"
  "scidtb"
#   "ud202xpos"
  )
# Generate all combinations
array_names=(
            seed
            use_gnn_steps_opts
            gnn_enc_layers_opts
            parser_type_opts
            top_k_opts
            arc_norm_opts
            gnn_dropout_opts
            gnn_activation_opts
            dataset_name_opts
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
training_steps=10000
eval_steps=500
results_suffix=gat

# Convert combinations to commands
declare -a commands=()
while IFS= read -r combo; do
    IFS=',' read -ra params <<< "$combo"
    
    cmd="python ./src/train.py
                --opts
                --results_suffix ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
                --seed ${params[0]}
                --use_gnn_steps ${params[1]}
                --gnn_enc_layers ${params[2]}
                --parser_type ${params[3]}
                --top_k ${params[4]}
                --arc_norm ${params[5]}
                --gnn_dropout ${params[6]}
                --gnn_activation ${params[7]}
                --dataset ${params[8]}
                --training_steps $training_steps 
                --eval_steps $eval_steps
                --use_tagger_rnn 0
                --use_parser_rnn 0
                --parser_rnn_hidden_size 400
                --parser_rnn_layers 0
                "
    if [ "${params[1]}" -gt 0 ] && [ "${parser_type}" == 'simple' ]; then
        continue
    fi
    commands+=("$cmd")
done <<< "$combinations"

total_combinations=${#commands[@]}

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    command_to_run="${commands[$SLURM_ARRAY_TASK_ID]}"
    echo "$command_to_run"
    $command_to_run
else
    echo "This script should be run as a SLURM array job."
    echo "Use: sbatch --array=0-$((total_combinations-1))%999 $0"
    echo "This will distribute $total_combinations jobs across N GPUs."
fi