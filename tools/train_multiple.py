"""
    A very simple training script!
"""

import torch
import os
from stepparser.utils import save_json, build_dataloader, get_label_index_mapping, setup_config, build_model, get_index_label_mapping, build_optimizer
from stepparser.model import StepParser
from stepparser.config import default_cfg
from stepparser.utils import ner_collate_fuction, train_epoch, run_evaluation
from stepparser.evaluation import evaluate_model
from tqdm import tqdm
from pprint import pprint
import copy, json, contextlib
import argparse

def train(config):
    ## modify config based on environment
    config = setup_config(config, {})
    save_json(config, os.path.join(config['save_dir'], 'config.json'))

    with open(os.path.join(config['save_dir'], 'log.txt'), 'w') as log_file:
        with contextlib.redirect_stdout(log_file):
            ## build a dataloader
            train_loader = build_dataloader(config, loader_type = 'train', collate_function=ner_collate_fuction)
            save_json(train_loader.dataset.label_index_map, os.path.join(config['save_dir'], 'labels.json'))

            ## build a validation dataloader
            val_loader = build_dataloader(config, loader_type = 'val', collate_function=ner_collate_fuction)

            ## build a model
            model = build_model(config)
            # model.load_state_dict(torch.load('saved_models/model_1_bert-base-multilingual-cased_20220725-172537/model.pth'))
            # model.eval()

            ## optimizer ## write build_optimizer function!
            optimizer = build_optimizer(config, model)

            ## save the model
            save_path = os.path.join(config['save_dir'], 'model.pth')

            ## train the model
            curr_val_result = 0.0
            latest_save = 0
            for epoch in tqdm(range(config['epochs'])):
                model = train_epoch(model,train_loader, optimizer, epoch)

                ## validation
                val_results = run_evaluation(model, val_loader, eval_function = evaluate_model, config = config, label_index_map=train_loader.dataset.label_index_map)

                ## early stopping if best model is already found!
                if latest_save < config['patience'] and config['early_stopping']:
                    if val_results['parser_labeled_results']['P'] > curr_val_result:
                        torch.save(model.state_dict(), save_path)    
                        curr_val_result = val_results['parser_labeled_results']['P']
                        latest_save = 0
                    else:
                        latest_save += 1

                    print(f'Current best precision for labeled evaluation is {curr_val_result}.')
                else: 
                    break
                
            model.load_state_dict(torch.load(save_path))
            val_results = run_evaluation(model, val_loader, eval_function = evaluate_model, config = config, label_index_map=train_loader.dataset.label_index_map)
            pprint(val_results)

            with open(os.path.join(config['save_dir'], 'metrics.json'), 'w') as metrics_file:
                json.dump(val_results, metrics_file)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=int, required=True)
    parser.add_argument('-e',type=int, required=True)
    args = parser.parse_args()

    experiments = [
        {}, #original
        {'seed': 7529},
        {'seed': 317},
        {'learning_rate': 0.002},
        {'learning_rate': 0.0005},
        {'betas': [0.9, 0.999]}, #recommended
        {'betas': [0.9, 0.99]},
        {'betas': [0.99, 0.9]},
        {'batch_size': 16},
        {'batch_size': 32},
    ]

    for i in range(args.s, args.e + 1):
        experiment = experiments[i]
        print('experiment {}: {}'.format(i, experiment))
        config = copy.deepcopy(default_cfg)
        config['save_dir'] = '_results/test/m'
        for key in experiment:
            config[key] = experiment[key]
        train(config)
