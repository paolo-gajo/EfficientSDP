import torch
import os
from model.utils import (save_json, 
                              build_dataloader,  
                              setup_config, 
                              build_model, 
                              get_args, 
                              set_seeds,
                              train_epoch,
                              run_evaluation,
                              load_json,
                              save_python_command,
                              save_reproduce_training_cmd)
from model.config import default_cfg, custom_config
from model.evaluation import evaluate_model
from model.utils.graph_data_utils import get_mappings
import numpy as np
import sys
import json
from copy import deepcopy
from tqdm.auto import tqdm
from logging import getLogger
import argparse
import time

def main(args):
    cfg_path = os.path.join(args.model_dir, 'config.json')
    config = json.load(open(cfg_path, 'r'))['config']
    print('Config:\n\n', json.dumps(config, indent=4))
    config['output_edge_scores'] = 1
    # Set seeds and show save directory
    set_seeds(config['seed'])

    inference_output_dir = './inference_outputs'
    model_name = config['save_dir'].split('/')[-1]
    model_outputs_dir = os.path.join(inference_output_dir, model_name)
    if not os.path.exists(model_outputs_dir):
        os.makedirs(model_outputs_dir)

    # Build dataloaders for training, validation, and testing
    train_loader = build_dataloader(config, loader_type='train')
    val_loader = build_dataloader(config, loader_type='val')
    test_loader = build_dataloader(config, loader_type='test')

    # Build label index map and set additional configurations
    all_splits_data = load_json(config['train_file_graphs']) + \
                    load_json(config['val_file_graphs']) + \
                    load_json(config['test_file_graphs'])
    label_index_map = get_mappings(all_splits_data)
    if config['procedural']:
        config['max_steps'] = max(train_loader.dataset.max_steps,
                              val_loader.dataset.max_steps,
                              test_loader.dataset.max_steps,)
    config['n_tags'] = len(label_index_map['tag2class'])
    config['n_edge_labels'] = len(label_index_map['edgelabel2class'])

    # Build model and set up optimizer
    model_start_path = config.get('model_start_path', None)
    model = build_model(config, model_start_path=model_start_path)

    model.eval()
    model.set_mode('test') ## this tells the model that we are testing it, so it will return us precision instead of loss
    outputs = []
    inputs = []

    ## let's time it and find average time
    times = []
    with torch.no_grad():
        with tqdm(test_loader, position=0, leave = False) as pbar:
            for inp_data in tqdm(test_loader, position=0, leave = False):
                inputs.extend(inp_data)
                st_time = time.time()
                outputs.extend(model(inp_data))
                tot_time = round((time.time() - st_time) / test_loader.batch_size, 3)
                pbar.set_description(f"Batch inference time is {tot_time} seconds", refresh = True)
                times.append(tot_time)
    output_file_path = os.path.join(model_outputs_dir, f"{model_name}_output.json")
    with open(output_file_path, 'w', encoding='utf8') as f:
        json.dump(outputs, f, ensure_ascii = False, indent=4)
    config_output_path = os.path.join(model_outputs_dir, 'config.json')
    with open(config_output_path, 'w', encoding='utf8') as f:
        json.dump(config, f, ensure_ascii = False, indent = 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run saved model inference.")
    parser.add_argument("--model_dir", help="Path of the config file.")
    args = parser.parse_args()
    main(args)