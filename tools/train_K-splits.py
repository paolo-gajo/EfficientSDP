"""
    Train K different splits of data on the model
    To build new splits and train the model, run the following command:
        CUDA_VISIBLE_DEVICES=1 python3 tools/train_K-splits.py --opts --splits "30" --seed "42" --train_file "data/Parser/train.conllu" --test_file "data/Parser/test.conllu" --dev_file "data/Parser/dev.conllu"
    If you already have splits, then run command below:
        CUDA_VISIBLE_DEVICES=1 python3 tools/train_K-splits.py --opts --splits "30" --seed "42" --train_file "data/Parser/train.conllu" --test_file "data/Parser/test.conllu" --dev_file "data/Parser/dev.conllu" --split_data_dir "/path/to/splits"
"""

import os

from model.utils import (make_dir,
                        get_args, 
                        read_conllu_dataset_allennlp,
                        get_current_time_string, 
                        write_text, 
                        is_file
                        )
import random
from math import floor

def get_train_test_val_split(input_data, train_test_val_split = [80, 10, 10], split_seed = None):
    """
        Train/test/val splits from input data
        train_test_val_split: The list must sum to 100
    """

    split_seed = split_seed if split_seed is not None else random.randint(0, 10000)
    random.Random(split_seed).shuffle(input_data)
    output_len = len(input_data)
    train_ind, test_ind, val_ind = floor(output_len * train_test_val_split[0] / 100), floor(output_len * (train_test_val_split[0] + train_test_val_split[1]) / 100), floor(output_len * (train_test_val_split[0]+train_test_val_split[1]+train_test_val_split[2]) / 100)
    train_data, test_data, val_data = input_data[0: train_ind], input_data[train_ind: test_ind], input_data[test_ind:]

    return train_data, test_data, val_data

def get_to_conllu_format(recipe_element):
    """
        This takes recipe element read using allennlp and put it in conllu format
    """
    indices = [i+1 for i in range(len(recipe_element['words'].tokens))]
    words = [token for token in recipe_element['words'].tokens]
    tags = [label for label in recipe_element['pos_tags'].labels]
    head_tags = [index for index in recipe_element['head_tags'].labels]
    head_indices = [index for index in recipe_element['head_indices'].labels]
    
    ## getting them in conllu format
    recipe_conllu = '\n'.join([f'{index}\t{token}\t_\t_\t{tag}\t_\t{head_index}\t{head_tag}\t_\t_' for index, token, tag, head_index, head_tag in zip(indices, words, tags, head_indices, head_tags)])

    return recipe_conllu

def build_train_test_val_splits(all_data, split_seed = None):
    """
        Build train, test and validation splits and return as a string to be stored in 
        same CoNLL-U format
    """
    train_data, test_data, val_data = get_train_test_val_split(all_data, split_seed=split_seed)

    ## get everything in CoNLL-U format 
    train_data = [get_to_conllu_format(input_elem) for input_elem in train_data]
    test_data = [get_to_conllu_format(input_elem) for input_elem in test_data]
    val_data = [get_to_conllu_format(input_elem) for input_elem in val_data]

    return '\n\n'.join(train_data), '\n\n'.join(test_data), '\n\n'.join(val_data)

## create k different splits
args = get_args()
splits = args['splits'] if 'splits' in args else 30
seed = args['seed'] if 'seed' in args else 42

assert "train_file" in args and "dev_file" in args and "test_file" in args, "Please provide paths to train, test and validation files"

## data_directory
date_time = get_current_time_string()
split_data_dir = os.path.join(f'/data/Multitask_RFG/{splits}-splits_{date_time}/') if 'split_data_dir' not in args else args['split_data_dir'] ## where K splits will be stored

## let's load the data
train_data, test_data, val_data = read_conllu_dataset_allennlp(args['train_file']), read_conllu_dataset_allennlp(args['test_file']), read_conllu_dataset_allennlp(args['dev_file'])
all_data = train_data + test_data + val_data

## this will have all train, test and validation paths
train_paths, test_paths, val_paths = [], [], []

## split seeds
split_seeds = random.sample(range(1, splits+1), splits)

## let's make split and save it
for split in range(splits):
    
    ## split seeds
    split_seed = split_seeds[split]
    
    split_dir = os.path.join(split_data_dir, f'split_{str(split).zfill(5)}')
    make_dir(split_dir)

    train_file, test_file, val_file = os.path.join(split_dir, 'train.conllu'), os.path.join(split_dir, 'test.conllu'), os.path.join(split_dir, 'dev.conllu')
    train_data, test_data, val_data = build_train_test_val_splits(all_data, split_seed)

    ## if files don't exist then we create them
    if not (is_file(train_file) and is_file(test_file) and is_file(val_file)):
        
        write_text(train_file, train_data)
        write_text(test_file, test_data)
        write_text(val_file, val_data)

    ## store all train/test/val paths
    train_paths.append(train_file)
    test_paths.append(test_file)
    val_paths.append(val_file)


## let's train
for i, (train_path, test_path, val_path) in enumerate(zip(train_paths, test_paths, val_paths)):

    save_dir = f"./results/{splits}_splits_finetune_with_new_softmax_on_old_split/"
    split = str(i).zfill(5)
    save_dir_split = os.path.join(save_dir, f'split_{split}')
    ## if training from scratch
    train_cmd = f'python3 tools/train.py --opts --train_file "{train_path}" --val_file "{val_path}" --test_file "{test_path}" --save_dir "{save_dir_split}" --seed "{seed}" --model_name "bert-base-uncased" --use_pred_tags "True" '

    ## if finetuning from a pretrained silver model
    # train_cmd = f'python3 tools/train.py --opts --train_file "{train_path}" --val_file "{val_path}" --test_file "{test_path}" --save_dir "{save_dir_split}" --seed "{seed}" --model_name "bert-base-uncased" --use_pred_tags "True" --model_start_path "results/Silver_data_pretraining/Recipe1_Silver_data_with_different_softmax_bert-base-uncased_2022-10-05--22:37:54/model.pth" --labels_json_path "results/Silver_data_pretraining/Recipe1_Silver_data_with_different_softmax_bert-base-uncased_2022-10-05--22:37:54/labels.json"'
    if os.system(train_cmd) != 0:
        print("Training failed, deleting {split_data_dir} and {save_dir} directories.")
        if 'split_data_dir' not in args:
            os.system(f'rm -r {split_data_dir} && rm -r {save_dir}')