#!/bin/bash
#SBATCH -J large
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=./.slurm/%A_%a_output.log
#SBATCH --error=./.slurm/%A_%a_error.log
#SBATCH --mem=64G
#SBATCH --array=0-N%999
mkdir -p .slurm
nvidia-smi
module load rust gcc arrow
source .env/bin/activate

# Define all parameter combinations
declare -a seed_values=(
  0
  1
  2
  3
  4
  )
declare -a dataset_name_options=(
  "ade"
  "conll04"
  "scierc"
  "yamakata"
  )
declare -a model_name_options=(
  # "answerdotai/ModernBERT-base"
  # "microsoft/deberta-v3-base"
  "microsoft/deberta-v3-large"
  # "bert-base-uncased"
  "bert-large-uncased"
  )
declare -a parser_type_options=(
  "simple"
  # "gnn"
  )
declare -a arc_norm_options=(
  0
  1
  )
declare -a gnn_enc_layers_options=(
  0
  # 1
  # 2
  # 3
  )
declare -a parser_residual_options=(0)
declare -a freeze_encoder_options=(
  # 1
  0
  )
declare -a use_lora_options=(
  0
  # 1
  )
declare -a use_tagger_rnn_options=(  # used to skip invalid combinations, cannot have both 0 and 1
  0
  # 1
  )
declare -a parser_rnn_type_options=(
  # "none"
  # "gru"
  "lstm"
  # "normlstm"
  )
parser_rnn_layers_options=(
  0
  1
  2
  3
  # 4
  # 5
  # 6
  # 7
  # 8
  # 9
  # 10
  )
parser_rnn_hidden_size_options=(
  # 0
  # 100
  # 200
  # 300
  400
  )
arc_representation_dim_options=(
  # 100
  # 300
  500
  )
tag_embedding_type_options=(
  "linear"
  # "embedding"
  # "none"
)
bias_type='simple'

# Fixed parameters
training='steps'
training_steps=10000
eval_steps=100
test_steps=100

results_suffix="_ft_base_10k_grad_clipping"

# new norm setting
# parser_init='xu+norm'
# bma_init='norm'

# original norm setting
parser_init='xu'
bma_init='xu'

use_warmup=0
warmup_ratio=0.06
scheduler_type='linear'

use_clip_grad_norm=1
grad_clip_norm=1.0

valid_combinations=()
for seed in "${seed_values[@]}"; do
  for parser_type in "${parser_type_options[@]}"; do
    for freeze_encoder in "${freeze_encoder_options[@]}"; do
      for gnn_enc_layers in "${gnn_enc_layers_options[@]}"; do
        for arc_norm in "${arc_norm_options[@]}"; do
          for use_tagger_rnn in "${use_tagger_rnn_options[@]}"; do
            for parser_rnn_type in "${parser_rnn_type_options[@]}"; do
              for model_name in "${model_name_options[@]}"; do
                for parser_residual in "${parser_residual_options[@]}"; do
                  for use_lora in "${use_lora_options[@]}"; do
                    for dataset_name in "${dataset_name_options[@]}"; do
                      for parser_rnn_layers in "${parser_rnn_layers_options[@]}"; do
                        for parser_rnn_hidden_size in "${parser_rnn_hidden_size_options[@]}"; do
                          for arc_representation_dim in "${arc_representation_dim_options[@]}"; do
                            for tag_embedding_type in "${tag_embedding_type_options[@]}"; do
                              if [ "$parser_rnn_layers" -gt 0 ] && [ "$freeze_encoder" == 0 ]; then
                                continue
                              fi
                              if [ "$use_tagger_rnn" == 1 ] && [ "$freeze_encoder" == 0 ]; then
                                continue
                              fi
                              valid_combinations+=("$seed $use_tagger_rnn $parser_type $freeze_encoder $gnn_enc_layers $arc_norm $parser_rnn_type $model_name $parser_residual $use_lora $dataset_name $parser_rnn_layers $parser_rnn_hidden_size $arc_representation_dim $tag_embedding_type")
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

# Get the total number of combinations
total_combinations=${#valid_combinations[@]}
echo "Total combinations: $total_combinations"

# If SLURM_ARRAY_TASK_ID exists, use it to select the combination
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -lt "$total_combinations" ]; then
    # Get the combination for this task
    current_combination=${valid_combinations[$SLURM_ARRAY_TASK_ID]}
    
    # Parse the combination
    # use_tag_embeddings_in_parser
    read -r seed use_tagger_rnn parser_type freeze_encoder gnn_enc_layers arc_norm \
    parser_rnn_type model_name parser_residual \
    use_lora dataset_name parser_rnn_layers parser_rnn_hidden_size arc_representation_dim tag_embedding_type <<< "$current_combination"
    
    # Run the command with these parameters
    command_to_run="python ./src/train.py --opt \
--seed $seed \
--use_tagger_rnn $use_tagger_rnn \
--parser_type $parser_type \
--freeze_encoder $freeze_encoder \
--gnn_enc_layers $gnn_enc_layers \
--arc_norm $arc_norm \
--parser_rnn_type $parser_rnn_type \
--model_name $model_name \
--parser_residual $parser_residual \
--use_lora $use_lora \
--dataset_name $dataset_name \
--parser_rnn_layers $parser_rnn_layers \
--parser_rnn_hidden_size $parser_rnn_hidden_size \
--arc_representation_dim $arc_representation_dim \
--tag_embedding_type $tag_embedding_type \
--bias_type $bias_type \
--parser_init $parser_init \
--bma_init $bma_init \
--results_suffix $results_suffix \
--training $training \
--training_steps $training_steps \
--eval_steps $eval_steps \
--use_warmup $use_warmup \
--warmup_ratio $warmup_ratio \
--scheduler_type $scheduler_type \
--test_steps $test_steps \
--use_clip_grad_norm $use_clip_grad_norm \
--grad_clip_norm $grad_clip_norm"
    
    echo "Running job $SLURM_ARRAY_TASK_ID: $command_to_run"
    $command_to_run
  else
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of range (0-$((total_combinations-1)))"
    exit 1
  fi
else
  # If run manually, print the total number of combinations
  echo "This script should be run as a SLURM array job."
  echo "Use: sbatch --array=0-$((total_combinations-1))%999 experiments.sh"
  echo "This will distribute $total_combinations jobs across N GPUs."
fi