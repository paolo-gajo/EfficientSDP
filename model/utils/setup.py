from typing import Dict
import torch
from model.utils import get_current_time_string, make_dir
from transformers import AutoConfig, set_seed
import os
import warnings
from pathlib import Path

def set_lr(config: Dict):
    model_name = config['model_name'].replace('/', '-').replace(' ', '')
    model_name = model_name if not config['use_abs_step_embeddings'] else 'step-bert'
    if 'large' in model_name:
        config['learning_rate'] = config['learning_rate_large']
    elif config['parser_rnn_type'] == 'transformer' or not config['freeze_encoder']:
        config['learning_rate'] = config['learning_rate_encoder']
    else:
        config['learning_rate'] = config['learning_rate_freeze']
    print(f"Learning rate: {config['learning_rate']}")
    return config

def set_save_dir(save_dir, save_suffix = '', default_save_dir = './results'):
    if not save_dir:
        save_dir = default_save_dir
        if save_suffix:
            save_dir = os.path.join(save_dir, save_suffix, get_current_time_string())
        else:
            save_dir = os.path.join(save_dir, get_current_time_string())
    
    if not os.path.exists(save_dir):
        make_dir(save_dir)
        print(f"Created dir: {save_dir}")
    else:
        print('results_dir already exists, is this a re-run?')
        print('make sure you are not overwriting inadvertedly!')
    return save_dir

def set_labels(config: Dict):
    if config['dataset_name'] in ['scidtb', 'enewt', 'UD_Arabic-PADT', 'UD_Chinese-GSD', 'UD_Italian-ISDT', 'UD_Japanese-GSD', 'UD_Spanish-AnCora', 'UD_Wolof-WTB',]:
        config['test_ignore_tag'] = []
        config['test_ignore_edges'] = []
        config['test_ignore_edge_dep'] = ['punct']
    elif config['dataset_name'] in ['ade', 'conll04', 'scierc']:
        config['test_ignore_tag'] = ['O', 'no_label']
        config['test_ignore_edges'] = ['0']
        config['test_ignore_edge_dep'] = ['root']
    elif config['dataset_name'] in ['erfgc']:
        config['test_ignore_tag'] = ['O', 'no_label']
        config['test_ignore_edges'] = ['0']
        config['test_ignore_edge_dep'] = ['root', '-']
    return config

def setup_config(config : Dict, args: Dict = {}, custom_config: Dict = {}) -> Dict:
    for key in custom_config:
        config[key] = custom_config[key]
    for key in args:
        if key not in config and key not in custom_config:
            warnings.warn(f'{key} is passed as an input but not a valid key in current config. So it is ignored while overriding config.')
        else:
            config[key] = args[key]

    config['save_dir'] = set_save_dir(config['save_dir'], config['save_suffix'], './results')
    config = set_lr(config)

    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['model_path'] = os.path.join(config['save_dir'], 'model.pth')
    
    config = set_labels(config)

    if config['task_type'] == 'nlp':
        for split in ['train', 'val', 'test']:
            config[f'{split}_file_graphs'] = config[f'{split}_file_graphs'].format(dataset_name = config['dataset_name'])
        if 'encoder_output_dim' not in args.keys():
            config['encoder_output_dim'] = AutoConfig.from_pretrained(config['model_name']).hidden_size
    
    set_seeds(config['seed'])

    config['top_k'] = int(config['top_k'])
    return config

def set_seeds(seed):
    # enable deterministic behavior
    set_seed(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False