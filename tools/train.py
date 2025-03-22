import torch
import os
from stepparser.utils import (save_json, 
                              build_dataloader,  
                              setup_config, 
                              build_model, 
                              make_dir,
                              get_args, 
                              set_seeds,
                              train_epoch,
                              run_evaluation,
                              load_json,
                              save_python_command,
                              save_reproduce_training_cmd)
from stepparser.model import StepParser
from stepparser.config import default_cfg, custom_config
from stepparser.evaluation import evaluate_model, evaluate_model_with_all_labels
from stepparser.utils.graph_data_utils import get_mappings, get_max_steps
from tqdm import tqdm
import numpy as np
from pprint import pprint
import sys
import json
from copy import deepcopy

# torch.set_printoptions(linewidth=100, threshold=100)

# Get the arguments to modify the config
args = get_args()

# Modify config based on environment and create save directory
config = setup_config(default_cfg, args=args, custom_config=custom_config)

make_dir(config['save_dir'])
print('Current config:\n\n', json.dumps(config, indent=4))

# Set seeds and display save directory
set_seeds(config['seed'])
print(f"Will save to: {config['save_dir']}")

# Build dataloaders for training, validation, and testing
train_loader = build_dataloader(config, loader_type='train')
val_loader = build_dataloader(config, loader_type='val')
test_loader = build_dataloader(config, loader_type='test')

# Build label index map from graph files
all_splits_data = load_json(config['train_file_graphs']) \
                + load_json(config['val_file_graphs']) \
                + load_json(config['test_file_graphs'])
label_index_map = get_mappings(all_splits_data)
config['max_steps'] = get_max_steps(all_splits_data)

config['n_tags'] = len(label_index_map['tag2class'])
config['n_edge_labels'] = len(label_index_map['edgelabel2class'])

# Build model (optionally initializing from a checkpoint for finetuning)
model_start_path = args['model_start_path'] if 'model_start_path' in args else None
model = build_model(config, model_start_path=model_start_path)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# Set up optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

# Initialize variables for tracking best model and early stopping
val_results_list = []
best_model_state = None
best_val_metric = -np.inf
patience_counter = 0

# Training loop
for epoch in range(1, config['epochs'] + 1):
    
    if epoch >= config['freeze_until_epoch']:
        model.encoder.unfreeze_encoder() 

    model = train_epoch(model, train_loader, optimizer, epoch)

    val_results, _ = run_evaluation(
        model,
        val_loader,
        eval_function=evaluate_model,
        config=config,
        label_index_map=label_index_map,
        epoch=epoch,
    )
    
    print(val_results)
    val_results_list.append(val_results)
    
    # Extract evaluation metrics
    parser_f1 = val_results['parser_labeled_results']['F1']
    tagger_f1 = val_results['tagger_results']['F1']
    labeled_f1 = tagger_f1 if config['freeze_parser'] else parser_f1

    # Update best model if the current metric is better
    if config['early_stopping']:
        if labeled_f1 > best_val_metric:
            print(f'Best model updated! ({labeled_f1} > {best_val_metric})')
            best_val_metric = labeled_f1
            # torch.save(model_state_dict, config['model_path'])
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping condition
        if patience_counter >= config['patience']:
            print("Early stopping triggered.")
            break

# Save the best model checkpoint if available and saving is enabled
if best_model_state is not None and config['save_model']:
    torch.save(best_model_state, config['model_path'])

# Save validation results and configuration details
save_json(val_results_list, os.path.join(config['save_dir'], "val_results.json"))
save_json(train_loader.dataset.label_index_map, os.path.join(config['save_dir'], 'labels.json'))

# Save the training command and reproduction details
cmd_file = os.path.join(config['save_dir'], 'train_command.txt')
save_python_command(cmd_file, sys.argv)
reproduce_training_cmd_file = os.path.join(config['save_dir'], 'full_train_reproduce_cmd.txt')
save_reproduce_training_cmd(sys.argv[0], config, args, reproduce_training_cmd_file)

if config['save_model']:
    # Final evaluation: load the best model and evaluate on validation and test sets
    # model_state_dict = torch.load(config['model_path'])
    model.load_state_dict(best_model_state)
    val_results, benchmark_metrics = run_evaluation(
        model,
        val_loader,
        eval_function=evaluate_model,
        config=config,
        label_index_map=train_loader.dataset.label_index_map
    )
    save_json(val_results, os.path.join(config['save_dir'], f"val_results_best_f1={val_results['parser_labeled_results']}.json"))
    save_json(benchmark_metrics, os.path.join(config['save_dir'], 'val_results_benchmark.json'))
    print(f'Val results:\n\n', val_results)

    test_results, benchmark_metrics = run_evaluation(
        model,
        test_loader,
        eval_function=evaluate_model,
        config=config,
        label_index_map=train_loader.dataset.label_index_map
    )
    save_json(test_results, os.path.join(config['save_dir'], f"test_results_f1={test_results['parser_labeled_results']}.json"))
    save_json(benchmark_metrics, os.path.join(config['save_dir'], 'test_results_benchmark.json'))
    print(f'Test results:\n\n', test_results)

summary = {'config': config, 'val_results': val_results, 'test_results': test_results}
save_json(summary, os.path.join(summary['config']['save_dir'], 'config.json'))
