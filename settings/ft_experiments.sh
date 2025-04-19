#!/bin/bash
#SBATCH -J train_array
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=./.slurm/%A_%a_output.log
#SBATCH --error=./.slurm/%A_%a_error.log
#SBATCH --mem=64G
#SBATCH --array=0-N%4  # Replace N with (total_jobs-1), %4 means run 4 jobs simultaneously

source ./.env/bin/activate

# Define all parameter combinations
declare -a seed_values=(0 1 2 3 4)
declare -a parser_type_options=("simple")
declare -a arc_norm_options=(0 1)
declare -a gnn_enc_layers_options=(
  0
  # 1
  )
declare -a use_tag_embeddings_in_parser_options=(0 1)
declare -a parser_residual_options=(
  0
  # 1
  )
declare -a freeze_encoder_options=(
    0
    # 1
    )
declare -a use_parser_rnn_options=(
    0
    # 1
    )
declare -a use_tagger_rnn_options=(
    0
    # 1
    )
declare -a use_lora_options=(
    0
    # 1
    )
declare -a parser_rnn_type_options=(
    "none"
    # "gru"
    # "lstm"
    )
declare -a model_name_options=(
  "bert-base-uncased"
  # "bert-large-uncased"
  )
declare -a dataset_name_options=(
  "ade"
  "conll04"
  "scierc"
  "yamakata"
)

# Fixed parameters
augment_train=0
augment_val=0
augment_test=0
augment_k_train=0
augment_k_val=0
augment_k_test=0
keep_og_train=1
keep_og_val=1
keep_og_test=1
use_bert_positional_embeddings=1
use_abs_step_embeddings=0
# use_tagger_rnn=0
use_gnn=0
use_step_mask=0
laplacian_pe=0
training='steps'
training_steps=2000
eval_steps=100

parser_rnn_layers=3
parser_rnn_hidden_size=400

# Generate all valid combinations
valid_combinations=()
for seed in "${seed_values[@]}"; do
  for parser_type in "${parser_type_options[@]}"; do
    for freeze_encoder in "${freeze_encoder_options[@]}"; do
      for gnn_enc_layers in "${gnn_enc_layers_options[@]}"; do
        for arc_norm in "${arc_norm_options[@]}"; do
          for use_parser_rnn in "${use_parser_rnn_options[@]}"; do
            for use_tagger_rnn in "${use_tagger_rnn_options[@]}"; do
              for use_tag_embeddings_in_parser in "${use_tag_embeddings_in_parser_options[@]}"; do
                for parser_rnn_type in "${parser_rnn_type_options[@]}"; do
                  for model_name in "${model_name_options[@]}"; do
                    for parser_residual in "${parser_residual_options[@]}"; do
                      for use_lora in "${use_lora_options[@]}"; do
                        for dataset_name in "${dataset_name_options[@]}"; do
                          if [ "$use_lora" == 1 ] && [ "$freeze_encoder" == 0 ]; then
                            continue
                          fi
                          if [ "$use_parser_rnn" == 1 ] && [ "$freeze_encoder" == 0 ]; then
                            continue
                          fi
                          if [ "$use_tagger_rnn" == 1 ] && [ "$freeze_encoder" == 0 ]; then
                            continue
                          fi
                          
                          # Add valid combination
                          valid_combinations+=("$seed $parser_type $freeze_encoder $gnn_enc_layers $arc_norm $use_parser_rnn $use_tag_embeddings_in_parser $parser_rnn_type $model_name $parser_residual $use_lora $dataset_name")
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

# for combo in "${valid_combinations[@]}"; do
#   echo "$combo"
#   echo ''
#   done

# If SLURM_ARRAY_TASK_ID exists, use it to select the combination
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -lt "$total_combinations" ]; then
    # Get the combination for this task
    current_combination=${valid_combinations[$SLURM_ARRAY_TASK_ID]}
    
    # Parse the combination
    read -r seed parser_type freeze_encoder gnn_enc_layers arc_norm use_parser_rnn \
         use_tag_embeddings_in_parser parser_rnn_type model_name parser_residual use_lora dataset_name <<< "$current_combination"
    
    # Run the command with these parameters
    command_to_run="python ./tools/train.py --opt \
--use_tag_embeddings_in_parser $use_tag_embeddings_in_parser \
--use_tagger_rnn $use_tagger_rnn \
--use_parser_rnn $use_parser_rnn \
--training $training \
--training_steps $training_steps \
--eval_steps $eval_steps \
--freeze_encoder $freeze_encoder \
--seed $seed \
--parser_type $parser_type \
--gnn_enc_layers $gnn_enc_layers \
--arc_norm $arc_norm \
--parser_rnn_type $parser_rnn_type \
--model_name $model_name \
--use_lora $use_lora \
--parser_residual $parser_residual \
--parser_rnn_layers $parser_rnn_layers \
--parser_rnn_hidden_size $parser_rnn_hidden_size \
--dataset_name $dataset_name"
    
    echo "Running job $SLURM_ARRAY_TASK_ID: $command_to_run"
    $command_to_run
  else
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of range (0-$((total_combinations-1)))"
    exit 1
  fi
else
  # If run manually, print the total number of combinations
  echo "This script should be run as a SLURM array job."
  echo "Use: sbatch --array=0-$((total_combinations-1))%4 your_script.sh"
  echo "This will distribute $total_combinations jobs across 4 GPUs."
fi