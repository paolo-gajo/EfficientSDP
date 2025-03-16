from torch.utils.data import DataLoader
from stepparser.utils.graph_data_utils import GraphCollator, GraphDataset
from transformers import AutoTokenizer
import random

def build_dataloader(config, loader_type = 'train'):
    collator = GraphCollator()
    kwargs = {'add_prefix_space': True}
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], **kwargs)
    
    if loader_type == 'train':
        train_data = GraphDataset.from_path(config['train_file_graphs'], tokenizer = tokenizer, split = 'train')
        if config['plot']:
            train_data.plot_topological_sorts_histogram(savename = './figures/train_hist.pdf')
        if config['only_use_biggest_graph'][loader_type]:
            train_data.only_use_max_step_graph(threshold=config['biggest_graph_threshold'])
        if config[f'augment_{loader_type}']:
            train_data.augment(k = config[f'augment_k_{loader_type}'], keep_og = config[f'keep_og_{loader_type}'])
        if config['shuffle'][loader_type]:
            print(f'Shuffling {loader_type} split...')
            train_data.shuffle()
        config['data_len'][loader_type] = len(train_data.data)
        train_loader = DataLoader(train_data, batch_size=config['batch_size'], collate_fn = collator.collate, shuffle = False)
        return train_loader

    if loader_type == 'val':
        val_data = GraphDataset.from_path(config['val_file_graphs'], tokenizer = tokenizer, split = 'val')
        if config['plot']:
            val_data.plot_topological_sorts_histogram(savename = './figures/val_hist.pdf')
        if config['only_use_biggest_graph'][loader_type]:
            val_data.only_use_max_step_graph(threshold=config['biggest_graph_threshold'])
        if config[f'augment_{loader_type}']:
            val_data.augment(k = config[f'augment_k_{loader_type}'], keep_og = config[f'keep_og_{loader_type}'])
        config['data_len'][loader_type] = len(val_data.data)
        val_loader = DataLoader(val_data, batch_size=config['batch_size'], collate_fn = collator.collate)
        return val_loader

    if loader_type == 'test':
        test_data = GraphDataset.from_path(config['test_file_graphs'], tokenizer = tokenizer, split = 'test')
        if config['plot']:
            test_data.plot_topological_sorts_histogram(savename = './figures/test_hist.pdf')
        if config[f'augment_{loader_type}']:
            test_data.augment(k = config[f'augment_k_{loader_type}'], keep_og = config[f'keep_og_{loader_type}'])
        config['data_len'][loader_type] = len(test_data.data)
        test_loader = DataLoader(test_data, batch_size=config['batch_size'], collate_fn = collator.collate)
        return test_loader
    return None