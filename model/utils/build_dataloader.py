import torch
from torch.utils.data import DataLoader
import torch_geometric as pyg
from model.utils.data_utils import TextGraphCollator, TextGraphDataset, GraphDataset
from transformers import AutoTokenizer
from model.utils import load_json
from model.utils.data_utils import get_mappings
from torch_geometric.datasets import QM9, TUDataset, AQSOL, MalNetTiny, GNNBenchmarkDataset, RelLinkPredDataset, ZINC, LRGBDataset
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
"zinc": "graph",
# "reddit": "graph",
"aqsol": "graph",
# "malnettiny": "graph",
"cifar10": "graph",
# "rlp": "graph",
"COIL-RAG": "graph",
"PROTEINS_full": "graph",
# "benzene": "graph",
"FRANKENSTEIN": "graph",
"PCQM-Contact": "graph",
}

GRAPH_DATASETS = {
    'qm9': lambda: QM9(root='data/QM9'),
    'zinc': {
        'train': lambda: ZINC(root='data/zinc', split='train'),
        'val': lambda: ZINC(root='data/zinc', split='val'),
        'test': lambda: ZINC(root='data/zinc', split='test'),
    },
    'aqsol': lambda: AQSOL(root='data/AQSOL'),
    'cifar10': lambda: GNNBenchmarkDataset(root='./data/CIFAR10_superpixel', name='CIFAR10'),
    'COIL-RAG': lambda: TUDataset(root='data', name='COIL-RAG', use_node_attr=True, use_edge_attr=True),
    'PROTEINS_full': lambda: TUDataset(root='data', name='PROTEINS_full', use_node_attr=True, use_edge_attr=True),
    'FRANKENSTEIN': lambda: TUDataset(root='data', name='FRANKENSTEIN', use_node_attr=True, use_edge_attr=True),
    'PCQM-Contact': {
        'train': lambda: LRGBDataset(root='data/PCQM-Contact', name='PCQM-Contact', split='train'),
        'val': lambda: LRGBDataset(root='data/PCQM-Contact', name='PCQM-Contact', split='val'),
        'test': lambda: LRGBDataset(root='data/PCQM-Contact', name='PCQM-Contact', split='test'),
    }
}

# Datasets that should have positional features concatenated
DATASETS_WITH_POS = ['cifar10', 'qm9']

# Datasets with predefined splits
PREDEFINED_SPLIT_DATASETS = ['PCQM-Contact', 'zinc']


def add_positional_features(dataset, dataset_name):
    """Add positional features to node features if available."""
    if dataset_name not in DATASETS_WITH_POS:
        return dataset
    new_data = []
    for graph in dataset:
        if hasattr(graph, 'pos') and graph.pos is not None:
            # Concatenate node features with positional coordinates
            graph.x = torch.cat([graph.x, graph.pos], dim=-1)
        new_data.append(graph)
    return new_data


def load_dataset_splits(dataset_name):
    """Load train/val/test splits for the given dataset."""
    if dataset_name in PREDEFINED_SPLIT_DATASETS:
        # Handle datasets with predefined splits (e.g., PCQM-Contact)
        splits = GRAPH_DATASETS[dataset_name]
        return splits['train'](), splits['val'](), splits['test']()
    else:
        # Load single dataset and split manually
        dataset = GRAPH_DATASETS[dataset_name]()
        return dataset, None, None


def split_dataset_manually(dataset, config):
    """Split a single dataset into train/val/test."""
    train_data, temp_data = train_test_split(
        list(dataset), test_size=0.3, random_state=config['seed']
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=config['seed']
    )
    return train_data, val_data, test_data


def update_config_dimensions(config, sample_data):
    """Update config with feature and edge dimensions from sample data."""
    # Set node feature dimension
    if sample_data.x is not None and sample_data.x.dim() > 1:
        config['feat_dim'] = sample_data.x.shape[-1]
    
    # Set edge feature dimension
    if sample_data.edge_attr is not None and sample_data.edge_attr.dim() > 1:
        config['edge_dim'] = sample_data.edge_attr.shape[-1]
    else:
        config['edge_dim'] = 1  # Default for geometric distance features


def create_data_loaders(train_data, val_data, test_data, config):
    """Create PyTorch Geometric data loaders."""
    # Limit evaluation samples if specified
    if config.get('eval_samples', 0):
        val_data = val_data[:config['eval_samples']]
        test_data = test_data[:config['eval_samples']]
    
    # Create datasets
    datasets = {
        'train': GraphDataset(train_data),
        'val': GraphDataset(val_data),
        'test': GraphDataset(test_data)
    }
    
    # Create data loaders
    loaders = {}
    for split, dataset in datasets.items():
        shuffle = (split == 'train')
        loaders[split] = pyg.loader.DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=shuffle
        )
    
    return loaders


def build_graph_dataloader(config):
    """Build graph data loaders for the specified dataset."""
    dataset_name = config['dataset_name']
    
    # Load dataset(s)
    train_dataset, val_dataset, test_dataset = load_dataset_splits(dataset_name)
    if train_dataset.x.dtype == torch.long:
        config['num_embeddings'] = int(torch.max(train_dataset.x) + 1)
    else:
        config['num_embeddings'] = 0
    config['num_node_feats'] = train_dataset.x.shape[-1]

    # Validate dataset
    assert len(train_dataset) > 1, f"Dataset {dataset_name} is too small"
    
    # Handle datasets that need manual splitting
    if dataset_name not in PREDEFINED_SPLIT_DATASETS:
        train_data, val_data, test_data = split_dataset_manually(train_dataset, config)
    else:
        train_data, val_data, test_data = list(train_dataset), list(val_dataset), list(test_dataset)
    
    # Add positional features if needed
    train_data = add_positional_features(train_data, dataset_name)
    val_data = add_positional_features(val_data, dataset_name)
    test_data = add_positional_features(test_data, dataset_name)
    
    # Update config with dataset dimensions
    update_config_dimensions(config, train_data[0])
    
    # Create and return data loaders
    data_loaders = create_data_loaders(train_data, val_data, test_data, config)
    return data_loaders


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