#!/bin/bash
#SBATCH -J gnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
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
    3
    4
)

declare -a lgi_enc_layers_opts=(
    # 0
    1
    2
    3
    # 4
    # 5
    )

declare -a arc_norm_opts=(
    0
    1
    )

declare -a epochs_opts=(
    # 1
    # 2
    3
)

declare -a arc_representation_dim_opts=(
    # 50
    100
    # 150
    # 200
)

declare -a encoder_output_dim_opts=(
    # 50
    100
    # 150
    # 200
)

declare -a dataset_name_opts=(
    # qm9
    cifar10
  )

array_names=(
            seed
            lgi_enc_layers_opts
            arc_norm_opts
            epochs_opts
            arc_representation_dim_opts
            encoder_output_dim_opts
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
eval_steps=10000
batch_size=64
save_suffix=lgi
learning_rate=0.001
task_type=graph
model_type=graph
use_clip_grad_norm=1
lgi_gat_type=base
gat_norm=0

declare -a commands=()
while IFS= read -r combo; do
    IFS=',' read -ra params <<< "$combo"

    cmd="python ./src/lgi_train.py
                --opts
                --save_suffix ${save_suffix}_${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
                --seed ${params[0]}
                --lgi_enc_layers ${params[1]}
                --arc_norm ${params[2]}
                --epochs ${params[3]}
                --arc_representation_dim ${params[4]}
                --encoder_output_dim ${params[5]}
                --dataset_name ${params[6]}
                --eval_steps $eval_steps
                --batch_size $batch_size
                --learning_rate $learning_rate
                --task_type $task_type
                --model_type $model_type
                --use_clip_grad_norm $use_clip_grad_norm
                --lgi_gat_type $lgi_gat_type
                --gat_norm $gat_norm
                "
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
        zero_index_num=$(( ${#commands[@]} - 1 ))
        echo "zero-index $((i)) of ${zero_index_num}"
        cmd="${commands[$i]} --save_suffix ${save_suffix}_${i}"
        echo "${cmd}"
        $cmd
    done
else
    echo "This script should be run as a SLURM array job."
    echo "Use: sbatch --array=0-$((total_combinations-1)) $0"
    echo "This will distribute $total_combinations jobs across N GPUs."
fi


