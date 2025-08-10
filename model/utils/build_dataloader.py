from torch.utils.data import DataLoader
import torch_geometric as pyg
from model.utils.data_utils import TextGraphCollator, TextGraphDataset, GraphDataset
from transformers import AutoTokenizer
from model.utils import load_json
from model.utils.data_utils import get_mappings
from torch_geometric.datasets import QM9, TUDataset
import json
import re
from sklearn.model_selection import train_test_split

DATASET_MAPPING = {"ade": "nlp",
"conll04": "nlp",
"DocRED": "nlp",
"enewt": "nlp",
"erfgc": "nlp",
"scidtb": "nlp",
"scierc": "nlp",
"UD_Arabic-PADT": "nlp",
"UD_Chinese-GSD": "nlp",
"UD_Italian-ISDT": "nlp",
"UD_Japanese-GSD": "nlp",
"UD_Spanish-AnCora": "nlp",
"UD_Wolof-WTB": "nlp",
"qm9": "graph",
}

GRAPH_DATASETS = {
    'qm9': QM9(root='data/QM9'),
    'reddit': TUDataset(root='data', name='REDDIT-BINARY')
}

def build_graph_dataloader(config):
    data = GRAPH_DATASETS[config['dataset_name']]
    config['feat_dim'] = data[0].x.shape[-1]
    config['edge_dim'] = data[0].edge_attr.shape[-1]
    config['n_edge_labels'] = 1
    data_train, intermediate = train_test_split(list(data), test_size=0.3, random_state=config['seed'])
    data_val, data_test = train_test_split(intermediate, test_size=0.5, random_state=config['seed'])
    dataset_train = GraphDataset(data_train)
    dataset_val = GraphDataset(data_val)
    if config['eval_samples']:
        dataset_val = dataset_val[:config['eval_samples']]
    dataset_test = GraphDataset(data_test)
    if config['eval_samples']:
        dataset_test = dataset_test[:config['eval_samples']]
    train_loader = pyg.loader.DataLoader(dataset_train,
                                batch_size=config['batch_size'],
                                shuffle = True)
    val_loader = pyg.loader.DataLoader(dataset_val,
                                batch_size=config['batch_size'],
                                shuffle = False)
    test_loader = pyg.loader.DataLoader(dataset_test,
                                batch_size=config['batch_size'],
                                shuffle = False)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }


def build_text_dataloader(config):
    collator = TextGraphCollator(config)
    kwargs = {'add_prefix_space': True}
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], **kwargs)
    
    all_splits_data = load_json(config['train_file_graphs']) + \
                    load_json(config['val_file_graphs']) + \
                    load_json(config['test_file_graphs'])
    
    train_data = json.load(open(config['train_file_graphs'], 'r'))
    val_data = json.load(open(config['val_file_graphs'], 'r'))
    test_data = json.load(open(config['test_file_graphs'], 'r'))

    # make a single label_index_map from all the data before using it for all datasets
    config['label_index_map'] = get_mappings(all_splits_data)
    config['n_tags'] = len(config['label_index_map']['tag2class'])
    config['n_edge_labels'] = len(config['label_index_map']['edgelabel2class'])

    for loader_type in ['train', 'val', 'test']:
        if loader_type == 'train':
            train_data = TextGraphDataset(train_data, config, split = 'train', tokenizer = tokenizer)
            config['data_len'][loader_type] = len(train_data.data)
            if config['procedural']:
                config['dataset_max_steps'][loader_type] = train_data.max_steps
            train_loader = DataLoader(train_data, batch_size=config['batch_size'], collate_fn = collator.collate, shuffle = False)

        if loader_type == 'val':
            val_data = TextGraphDataset(val_data, config, split = 'val', tokenizer = tokenizer)
            config['data_len'][loader_type] = len(val_data.data)
            if config['procedural']:
                config['dataset_max_steps'][loader_type] = val_data.max_steps
            val_loader = DataLoader(val_data, batch_size=config['batch_size'], collate_fn = collator.collate)

        if loader_type == 'test':
            test_data = TextGraphDataset(test_data, config, split = 'test', tokenizer = tokenizer)
            config['data_len'][loader_type] = len(test_data.data)
            if config['procedural']:
                config['dataset_max_steps'][loader_type] = test_data.max_steps
            test_loader = DataLoader(test_data, batch_size=config['batch_size'], collate_fn = collator.collate)

    if config['procedural']:
        config['max_steps'] = max(train_loader.dataset.max_steps,
                              val_loader.dataset.max_steps,
                              test_loader.dataset.max_steps,)
        
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }

def build_dataloader(config):
    assert config['task_type'] == DATASET_MAPPING[config['dataset_name']]
    if config['task_type'] == 'nlp':
        return build_text_dataloader(config)
    else:
        return build_graph_dataloader(config)