from torch.utils.data import DataLoader
from model.utils.graph_data_utils import GraphCollator, GraphDataset
from transformers import AutoTokenizer
import random

def build_dataloader(config, loader_type = 'train'):
    collator = GraphCollator()
    kwargs = {'add_prefix_space': True}
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], **kwargs)
    
    if loader_type == 'train':
        train_data = GraphDataset.from_path(config, split = 'train', tokenizer = tokenizer)
        config['data_len'][loader_type] = len(train_data.data)
        config['dataset_max_steps'][loader_type] = train_data.max_steps
        train_loader = DataLoader(train_data, batch_size=config['batch_size'], collate_fn = collator.collate, shuffle = False)
        return train_loader

    if loader_type == 'val':
        val_data = GraphDataset.from_path(config, split = 'val', tokenizer = tokenizer)
        config['data_len'][loader_type] = len(val_data.data)
        config['dataset_max_steps'][loader_type] = val_data.max_steps
        val_loader = DataLoader(val_data, batch_size=config['batch_size'], collate_fn = collator.collate)
        return val_loader

    if loader_type == 'test':
        test_data = GraphDataset.from_path(config, split = 'test', tokenizer = tokenizer)
        config['data_len'][loader_type] = len(test_data.data)
        config['dataset_max_steps'][loader_type] = test_data.max_steps
        test_loader = DataLoader(test_data, batch_size=config['batch_size'], collate_fn = collator.collate)
        return test_loader
    return None