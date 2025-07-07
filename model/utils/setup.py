from typing import Dict
import torch
from model.utils import get_current_time_string, make_dir
from transformers import AutoConfig, set_seed
import os
import warnings
from pathlib import Path

def setup_config(config : Dict, args: Dict = {}, custom_config: Dict = {}, mode = 'train') -> Dict:
    for key in custom_config:
        config[key] = custom_config[key]
    for key in args:
        if key not in config and key not in custom_config:
            warnings.warn(f'{key} is passed as an input but not a valid key in current config. So it is ignored while overriding config.')
        else:
            config[key] = args[key]

    # when we are doing validation or test, we just need to change variables
    # from command line args
    if mode == 'validation' or mode == 'test':
        return config

    # let's set the device correctly
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    splits = ['train', 'val', 'test']
    # correct directory name
    aug_string = ''.join([str(el) for el in [config[f'augment_{split}'] for split in splits]])
    k_string = '-'.join([str(el) for el in [config[f'augment_k_{split}'] for split in splits]])
    keep_og_string = ''.join([str(el) for el in [config[f'keep_og_{split}'] for split in splits]])
    keep_k_string = f"keep_{keep_og_string}_k_{k_string}"
    main_save_dir, model_name = config['save_dir'], config['model_name'].replace('/', '-').replace(' ', '')
    model_name = model_name if not config['use_abs_step_embeddings'] else 'step-bert'
    save_path = os.path.join(f"{main_save_dir}",
                            f"{config['results_suffix']}",
                            )
    
    if config['dataset_name'] in ['enewt', 'scidtb']:
        config['test_ignore_edge_dep'] = ['punct']
        config['test_ignore_tag'] = []
        config['test_ignore_edges'] = []
        
    if config['freeze_encoder']:
        config['learning_rate'] = config['learning_rate_freeze']
    else:
        config['learning_rate'] = config['learning_rate_encoder']
    if 'large' in model_name:
        config['learning_rate'] = config['learning_rate_large']

    config['save_dir'] = save_path
    print(f'Created dir: {save_path}')
    make_dir(config['save_dir'])
    config['figures_dir'] = f'./paper/figures_{keep_k_string}'
    make_dir(config['figures_dir'])

    config['train_file_graphs'] = config['train_file_graphs'].format(dataset_name = config['dataset_name'])
    config['val_file_graphs'] = config['val_file_graphs'].format(dataset_name = config['dataset_name'])
    config['test_file_graphs'] = config['test_file_graphs'].format(dataset_name = config['dataset_name'])

    # model path
    if Path(save_path) != Path(main_save_dir):
        config['model_path'] = os.path.join(config['save_dir'], 'model.pth')
    else:
        print(f'Save path is the same as main save dir `{main_save_dir}`!')

    # get encoder output dimension (aka hidden size)
    config['encoder_output_dim'] = AutoConfig.from_pretrained(config['model_name']).hidden_size
    
    set_seeds(config['seed'])

    config['top_k'] = int(config['top_k'])

    return config

def set_seeds(seed):
    """
        Enable deterministic behavior.
        https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py#L58
    """
    set_seed(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False