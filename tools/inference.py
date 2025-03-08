"""
    This is to run inference on new data and save it as a json file. 
    You can run this script in a few ways. You can provide list of recipes in the command line as below

    Way 1: here you can directly provide recipes using command line as shown below, to run your inference.
        CUDA_VISIBLE_DEVICES=2 python3 tools/inference.py --opts --dir_name "/path/to/model/directory" --recipes "["text of recipe 1", "text of recipe 2"]" --output_file_path "/path/to/output/file/in/conllu/format" --use_pred_tags "True"

    Way 2: In this way, you can provide recipes in a text file, where each line is interepreted as a single recipe.
        CUDA_VISIBLE_DEVICES=2 python3 tools/inference.py --opts --dir_name "/path/to/model/directory" --recipes_file_path "/path/to/recipe/txt/file" --output_file_path "/path/to/output/file/in/conllu/format" --use_pred_tags "True"
    
    Way 3: In this one, you do inference directly on CoNLLU file! 
        CUDA_VISIBLE_DEVICES=2 python3 tools/inference.py --opts --dir_name "/path/to/model/directory" --conllu_test_path "/path/to/recipe/conllu/file" --output_file_path "/path/to/output/file/in/conllu/format" --use_pred_tags "True"

"""

import os
from stepparser.utils import (build_dataloader, 
                        build_model, 
                        get_args, 
                        load_json, 
                        read_text,
                        setup_config, 
                        ner_collate_fuction,
                        is_file, 
                        run_inference,
                        remove,
                        build_conllu_file_from_recipe_list,
                        write_text,
                        plot_from_conllu,
                        is_arborescence)
from stepparser.config import default_cfg
import sys

## get arguments from the command lind
args = get_args()

## temporary save file for recipes in conllu format
args['test_file'] = '/tmp/tmp.conllu'

## directory in which model is location, dir path must be given!
dir_name = args['dir_name'] 

## save file name if results are to be saved
output_file_path = args['output_file_path'] if 'output_file_path' in args else '/tmp/output.conllu'

## load the config
saved_config = load_json(os.path.join(dir_name, 'config.json'))

## merge loaded config with default config
config = setup_config(default_cfg, saved_config, mode = 'test')

## merge args with loaded config
config = setup_config(config, args, mode = 'test')

## loading saved model
config['model_path'] = os.path.join(dir_name, 'model.pth')

## during inference/test, we don't shuffle the data and use full data
config['shuffle'] = False
config['fraction_dataset'] = 1.1

## always use predicted tags during inference
config['use_pred_tags'] = True

## build/set the input file
if 'recipes' in args:
    recipes = args['recipes']
    build_conllu_file_from_recipe_list(recipes, config['test_file'])
elif 'recipes_file_path' in args:
    recipes = read_text(args['recipes_file_path'])
    build_conllu_file_from_recipe_list(recipes, config['test_file'])
elif 'conllu_test_path' in args:
    config['test_file'] = args['conllu_test_path']
    assert is_file(config['test_file']), f'{config["test_file"]} is not a valid file, please provide valie conllu file for inference.'
else:
    print("No valid input for inference provided, terminating inference.")
    sys.exit(0)

## load presaved labels
label_index_map = load_json(os.path.join(dir_name, 'labels.json')) ## this will be used to get the actual labels!

## build dataloader
test_dataloader = build_dataloader(config, loader_type = 'test', collate_function = ner_collate_fuction, label_index_map=label_index_map)

## let's get the model! 
model = build_model(config)

## run inference and save the results!
output = run_inference(model, test_dataloader, config = config, label_index_map = label_index_map)
write_text(output_file_path, output)
print(f'Inference results are saved at {output_file_path}.')

## remove temporary files
if 'conllu_test_path' not in args:
    remove(config['test_file']) 

## let's check if all the outputs are trees or not
are_trees = is_arborescence(output_file_path)

## we also plot the graphs and save them
save_dir = os.path.join(dir_name, f'output_graphs')
print(f'Creating plots and saving it to {save_dir} directory!')
plot_from_conllu(output_file = output_file_path, save_dir = save_dir)