"""
    Let's do the evaluation and inference!
    CUDA_VISIBLE_DEVICE=2 python3 tools/evaluate.py --opts --dir_name "/path/to/model/directory" --test_file /path/to/test/file/in/conllu/format --batch_size "16" --save_file_name "test_results.json" --use_pred_tags "True"
"""

import os
from stepparser.utils import (build_dataloader, 
                        build_model, 
                        get_args, 
                        load_json, 
                        save_json,
                        setup_config, 
                        ner_collate_fuction, 
                        run_evaluation,
                        pretty_print_json)
from stepparser.evaluation import evaluate_model
from stepparser.config import default_cfg
from pprint import pprint

## get arguments from the command lind
args = get_args()

## directory in which model is location, dir path must be given!
dir_name = args['dir_name'] 

## save file name if results are to be saved
save_file_name = args['save_file_name'] if 'save_file_name' in args else None

## load the config
saved_config = load_json(os.path.join(dir_name, 'config.json'))

## merge loaded config with default config
config = setup_config(default_cfg, saved_config, mode = 'test')

## merge args with loaded config
config = setup_config(config, args, mode = 'test')

## model_path
config['model_path'] = os.path.join(dir_name, 'model.pth')
 
## during inference/test, we don't shuffle the data and use full data
config['shuffle'] = False
config['fraction_dataset'] = 1.1

## use predicted tags for evaluation unless provided otherwise
config['use_pred_tags'] = True if not 'use_pred_tags' in args else args['use_pred_tags']

## load presaved labels
label_index_map = load_json(os.path.join(dir_name, 'labels.json')) ## this will be used to get the actual labels!

## build dataloader
test_dataloader = build_dataloader(config, loader_type = 'test', collate_function = ner_collate_fuction, label_index_map=label_index_map)

## get the model
model = build_model(config)

## let's run the evaluation!
test_results, benchmark_metrics = run_evaluation(model, test_dataloader, eval_function = evaluate_model, config = config, label_index_map = label_index_map)
pretty_print_json(test_results, string = 'Test results: ')
pretty_print_json(benchmark_metrics, string = 'Performance benchmark: ')

## save file if asked for
if save_file_name is not None:
    save_file_path = os.path.join(dir_name, save_file_name)
    save_json(test_results, save_file_path)