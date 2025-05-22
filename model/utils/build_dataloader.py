from torch.utils.data import DataLoader
from model.utils.graph_data_utils import GraphCollator, GraphDataset
from transformers import AutoTokenizer
from model.utils import load_json
from model.utils.graph_data_utils import get_mappings

def build_dataloader(config):
    collator = GraphCollator(config)
    kwargs = {'add_prefix_space': True}
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], **kwargs)
    
    all_splits_data = load_json(config['train_file_graphs']) + \
                    load_json(config['val_file_graphs']) + \
                    load_json(config['test_file_graphs'])
    
    # make a single label_index_map from all the data before using it for all datasets
    config['label_index_map'] = get_mappings(all_splits_data)
    config['n_tags'] = len(config['label_index_map']['tag2class'])
    config['n_edge_labels'] = len(config['label_index_map']['edgelabel2class'])

    for loader_type in ['train', 'val', 'test']:
        if loader_type == 'train':
            train_data = GraphDataset.from_path(config, split = 'train', tokenizer = tokenizer)
            config['data_len'][loader_type] = len(train_data.data)
            if config['procedural']:
                config['dataset_max_steps'][loader_type] = train_data.max_steps
            train_loader = DataLoader(train_data, batch_size=config['batch_size'], collate_fn = collator.collate, shuffle = False)

        if loader_type == 'val':
            val_data = GraphDataset.from_path(config, split = 'val', tokenizer = tokenizer)
            config['data_len'][loader_type] = len(val_data.data)
            if config['procedural']:
                config['dataset_max_steps'][loader_type] = val_data.max_steps
            val_loader = DataLoader(val_data, batch_size=config['batch_size'], collate_fn = collator.collate)

        if loader_type == 'test':
            test_data = GraphDataset.from_path(config, split = 'test', tokenizer = tokenizer)
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