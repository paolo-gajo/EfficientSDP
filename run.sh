#!/bin/bash
#SBATCH -J train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log

# parser_opts=('mtrfg' 'gnn')

source .env/bin/activate

python ./tools/train.py

# for parser_type in "${parser_opts[@]}"
# do
#     python ./tools/train.py --opts \
#     --parser_type $parser_type
# done