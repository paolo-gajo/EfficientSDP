import torch
import os
from stepparser.utils import (save_json, 
                        build_dataloader,  
                        setup_config, 
                        build_model, 
                        # build_optimizer, 
                        make_dir,
                        get_args, 
                        set_seeds,
                        # write_text,
                        # ner_collate_fuction,
                        train_epoch,
                        run_evaluation,
                        load_json,
                        validate_epoch,
                        # get_label_index_mapping,
                        save_python_command,
                        save_reproduce_training_cmd
                        )

from stepparser.model import StepParser
from stepparser.config import default_cfg, custom_config
from stepparser.evaluation import evaluate_model, evaluate_model_with_all_labels
from stepparser.utils.graph_data_utils import get_mappings
from tqdm import tqdm
import numpy as np
from pprint import pprint
import sys
import json
torch.set_printoptions(linewidth=100000, threshold=100000)

## get the arguments to modify the config
args = get_args()

## modify config based on environment
config = setup_config(default_cfg,
                      args = args,
                      custom_config = custom_config
                      )

print('Current custom config:\n\n', json.dumps(custom_config, indent = 4))
## set seeds
set_seeds(config['seed'])
print(f"Will save to: {config['save_dir']}")

## build a dataloader
train_loader = build_dataloader(config, loader_type = 'train')
val_loader = build_dataloader(config, loader_type = 'val')
test_loader = build_dataloader(config, loader_type = 'test')

label_index_map = get_mappings(load_json(config['train_file_graphs']) + load_json(config['val_file_graphs']) + load_json(config['test_file_graphs']))

config['n_tags'] = len(label_index_map['tag2class'])
config['n_edge_labels'] = len(label_index_map['edgelabel2class'])

## build a model
model_start_path = args['model_start_path'] if 'model_start_path' in args else None ## this is an extra argument introduced that'd perform model initialization for finetuning!
"""
This is to load the model from a certain checkpoint, could be used to resume the training or finetuning a new model. 
Note that when you are performing finetuning, pass --labels_json_path argument with appropriate labels file
so that model is finetuned/continued being trained on the same labels it saw earlier! 
"""
model = build_model(config, model_start_path = model_start_path)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

## optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr = config['learning_rate'])

## train the model 
curr_best_val_value = -np.inf
latest_save = 0
val_results_list = []
with tqdm(range(1, config['epochs'] + 1)) as pbar:
    for epoch in pbar:

        if epoch >= config['freeze_until_epoch']:
            model.encoder.unfreeze_encoder() 

        model = train_epoch(model, train_loader, optimizer, epoch)

        ## validation
        ## let's evaluate model on validation dataset
        eval_function = evaluate_model
        eval_function_name = eval_function.__name__
        val_results, _ = run_evaluation(model,
                                        val_loader,
                                        eval_function = eval_function,
                                        config = config,
                                        label_index_map = label_index_map,
                                        epoch = epoch,
                                        )
        print(val_results)
        val_results_list.append(val_results)
        parser_prec = val_results['parser_labeled_results']['P']
        tagger_prec = val_results['tagger_results']['P']

        ## let's get validation loss for early stopping
        # val_loss = validate_epoch(model, val_loader)

        ## if parser is frozen, then tagger result should determine early stopping
        labeled_prec = tagger_prec if config['freeze_parser'] else parser_prec

        ## early stopping if best model is already found!
        if latest_save < config['patience'] and config['early_stopping']:
            if labeled_prec > curr_best_val_value:
                if config['save_model']:
                    torch.save(model.state_dict(), config['model_path'])    
                curr_best_val_value = round(labeled_prec, 3)
                
                latest_save = 0
            else:
                latest_save += 1

            # print(f'Epoch: {epoch}, labeled precision: {labeled_prec}')
        else: 
            break

## make directory and save config!
f1_string = str(round(val_results['parser_labeled_results']['F1'], 3))
config['save_dir'] = f"{config['save_dir']}_f1={f1_string}"
make_dir(config['save_dir'])
save_json(val_results_list, os.path.join(config['save_dir'], f"val_results.json"))
save_json(config, os.path.join(config['save_dir'], 'config.json'))
config.update(val_results_list[-1])
save_json(train_loader.dataset.label_index_map, os.path.join(config['save_dir'], 'labels.json'))
## command run should be saved so we know what did we run exactly to get that training
cmd_file = os.path.join(config['save_dir'], 'train_command.txt')
save_python_command(cmd_file, sys.argv)
## build a command to reproduce exact same training! 
reproduce_training_cmd_file = os.path.join(config['save_dir'], 'full_train_reproduce_cmd.txt')
save_reproduce_training_cmd(sys.argv[0], config, args, reproduce_training_cmd_file)

if config['save_model']:
    ## training is done, let's run the final evaluation
    model.load_state_dict(torch.load(config['model_path']))
    val_results, benchmark_metrics = run_evaluation(model, val_loader, eval_function = evaluate_model, config = config, label_index_map = train_loader.dataset.label_index_map)
    save_json(val_results, os.path.join(config['save_dir'], 'val_results_best.json'))
    save_json(benchmark_metrics, os.path.join(config['save_dir'], 'val_results_benchmark.json'))
    pprint(val_results)

    ## let's run the trained model on test dataset
    ## let's run and get results on test dataset too
    test_results, benchmark_metrics = run_evaluation(model, test_loader, eval_function = evaluate_model, config = config, label_index_map = train_loader.dataset.label_index_map)
    save_json(test_results, os.path.join(config['save_dir'], 'test_results.json'))
    save_json(benchmark_metrics, os.path.join(config['save_dir'], 'test_results_benchmark.json'))