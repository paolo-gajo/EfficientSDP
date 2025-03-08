"""
    Functions to modify config based on parameters of current environment
"""

from typing import Dict
import torch
from stepparser.utils import get_current_time_string
from transformers import AutoConfig, set_seed
import os
import numpy
import random
import warnings


def setup_config(config : Dict, args: Dict = {}, custom_config: Dict = {}, mode = 'train') -> Dict:
    """
        modify config params here before storing them
    """

    """
        modify config
        from arguments based on command line
    """
    
    for key in args:
        if key not in config:
            warnings.warn(f'{key} is passed as an input but not a valid key in current config. So it is ignored while overriding config.')
        else:
            config[key] = args[key]

    for key in custom_config:
        config[key] = custom_config[key]


    ## when we are doing validation or test, we just need to change variables
    ## from command line args
    if mode == 'validation' or mode == 'test':
        return config

    ## let's set the device correctly
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    ## correct directory name
    aug_string = ''.join([str(el) for el in config['augment_splits'].values()])
    save_dir, model_name = config['save_dir'], config['model_name'].replace('/', '-').replace(' ', '')
    dir_path = os.path.join(save_dir, f"{model_name}_{get_current_time_string()}_aug={aug_string}")
    config['save_dir'] = dir_path

    ## model path
    config['model_path'] = os.path.join(config['save_dir'], 'model.pth')

    ## get encoder output dimension (aka hidden size)
    config['encoder_output_dim'] = AutoConfig.from_pretrained(config['model_name']).hidden_size

    return config

def set_seeds(seed):

    """
        This is to enable fully deterministic behaviour.
        Borrowed from 
        https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py#L58
    """
    set_seed(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False