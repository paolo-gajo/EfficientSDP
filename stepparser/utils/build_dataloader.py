from torch.utils.data import DataLoader
from stepparser.utils.graph_data_utils import GraphCollator, GraphDataset
from transformers import AutoTokenizer
import random

def build_dataloader(config, loader_type = 'train'):
    augment_k = config['augment_k']
    keep_og = config['keep_og']
    collator = GraphCollator()
    kwargs = {'add_prefix_space': True}
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], **kwargs)
    
    if loader_type == 'train':
        train_data = GraphDataset.from_path(config['train_file_graphs'], tokenizer = tokenizer, split = 'train')
        if config['only_use_biggest_graph'][loader_type]:
            train_data.only_use_max_step_graph(threshold=config['biggest_graph_threshold'])
        if config['augment_splits'][loader_type]:
            train_data.augment(k = augment_k[loader_type], keep_og = keep_og)
        if config['shuffle'][loader_type]:
            print(f'Shuffling {loader_type} split...')
            train_data.shuffle()
        train_loader = DataLoader(train_data, batch_size=config['batch_size'], collate_fn = collator.collate, shuffle = False)
        return train_loader

    if loader_type == 'val':
        val_data = GraphDataset.from_path(config['val_file_graphs'], tokenizer = tokenizer, split = 'val')
        if config['only_use_biggest_graph'][loader_type]:
            val_data.only_use_max_step_graph(threshold=config['biggest_graph_threshold'])
        if config['augment_splits'][loader_type]:
            val_data.augment(k = augment_k[loader_type], keep_og = keep_og)
        val_loader = DataLoader(val_data, batch_size=config['batch_size'], collate_fn = collator.collate)
        return val_loader

    if loader_type == 'test':
        test_data = GraphDataset.from_path(config['test_file_graphs'], tokenizer = tokenizer, split = 'test')
        if config['augment_splits'][loader_type]:
            test_data.augment(k = augment_k[loader_type], keep_og = keep_og)
        test_loader = DataLoader(test_data, batch_size=config['batch_size'], collate_fn = collator.collate)
        return test_loader

    return None