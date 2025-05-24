#!/bin/bash
#SBATCH -J gnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log
#SBATCH --mem=64g
mkdir -p .slurm
nvidia-smi
module load rust gcc arrow
. .env/bin/activate

declare -a use_gnn_steps_opts=(
    1000
    -1
    )
declare -a gnn_enc_layers_opts=(
    3
    0
    )
declare -a parser_type_opts=(
    # simple
    # gat_unbatched
    gat
    )
top_k=1

for use_gnn_steps in "${use_gnn_steps_opts[@]}"
do
    for gnn_enc_layers in "${gnn_enc_layers_opts[@]}"
    do
        for parser_type in "${parser_type_opts[@]}"
        do
            cmd="python ./src/train.py
                        --opts --parser_type $parser_type
                        --training_steps 20000
                        --use_gnn_steps $use_gnn_steps
                        --gnn_enc_layers $gnn_enc_layers
                        --results_suffix _usegnn_${use_gnn_steps}_${top_k}
                        --eval_steps 500
                        --test_steps 500
                        --use_tagger_rnn 0
                        --use_parser_rnn 0
                        --parser_rnn_hidden_size 400
                        --parser_rnn_layers 0
                        --top_k $top_k"
            echo $cmd
            # $cmd
        done
    done
done

