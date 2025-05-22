from typing import Dict
import torch
from model.utils import get_current_time_string, make_dir
from transformers import AutoConfig, set_seed
import os
import warnings

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
    tagger_rnn_string = 1 if config['use_tagger_rnn'] else 0
    parser_rnn_string = 1 if config['use_parser_rnn'] else 0
    save_dir, model_name = config['save_dir'], config['model_name'].replace('/', '-').replace(' ', '')
    augment_type = 'none' if '1' not in aug_string else config['augment_type']
    model_name = model_name if not config['use_abs_step_embeddings'] else 'step-bert'
    parser_type = 'mtrfg' if config['parser_type'] == 'mtrfg' else f"{config['parser_type']}_{config['gnn_enc_layers']}"
    dir_path = os.path.join(f"{save_dir}{config['results_suffix']}",
                            f"freeze_encoder_{config['freeze_encoder']}",
                            f"arc_pred{config['arc_pred']}",
                            # f"stepmask_{config['use_step_mask']}",
                            # f"bpos_{config['use_bert_positional_embeddings']}",
                            f'tagger_rnn_{tagger_rnn_string}',
                            f"parser_rnn_{parser_rnn_string}_{config['parser_rnn_type']}_l{config['parser_rnn_layers']}_h{config['parser_rnn_hidden_size']}",
                            # f"laplacian_pe_{config['laplacian_pe']}",
                            # f"use_abs_step_embeddings_{config['use_abs_step_embeddings']}",
                            f"data={config['dataset_name']}",
                            f"parser_type_{parser_type}_mlp_{config['arc_representation_dim']}",
                            f"arc_norm_{config['arc_norm']}",
                            f"use_lora_{config['use_lora']}",
                            f"tag_embedding_type_{config['tag_embedding_type']}",
                            f"{model_name}_{get_current_time_string()}_seed_{config['seed']}",
                            )
    if config['dataset_name'] in ['ud202xpos', 'scidtb']:
        config['test_ignore_edge_dep'] = ['punct']
        config['test_ignore_tag'] = []
        config['test_ignore_edges'] = []
        
    if config['freeze_encoder']:
        config['learning_rate'] = config['learning_rate_freeze']
    else:
        config['learning_rate'] = config['learning_rate_encoder']
    if 'large' in model_name:
        config['learning_rate'] = config['learning_rate_large']

    print('learning rate:', config['learning_rate'])

    config['save_dir'] = dir_path
    print(f'Created dir: {dir_path}')
    make_dir(config['save_dir'])
    config['figures_dir'] = f'./paper/figures_{keep_k_string}'
    make_dir(config['figures_dir'])

    config['train_file_graphs'] = config['train_file_graphs'].format(dataset_name = config['dataset_name'])
    config['val_file_graphs'] = config['val_file_graphs'].format(dataset_name = config['dataset_name'])
    config['test_file_graphs'] = config['test_file_graphs'].format(dataset_name = config['dataset_name'])

    # model path
    config['model_path'] = os.path.join(config['save_dir'], 'model.pth')

    # get encoder output dimension (aka hidden size)
    config['encoder_output_dim'] = AutoConfig.from_pretrained(config['model_name']).hidden_size
    
    set_seeds(config['seed'])

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