from torch.utils.data import DataLoader
from model.utils.data_utils import GraphCollator, GraphDataset
from transformers import AutoTokenizer
from model.utils import load_json
from model.utils.data_utils import get_mappings
import json
import re

def build_dataloader(config):
    collator = GraphCollator(config)
    kwargs = {'add_prefix_space': True}
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], **kwargs)
    
    all_splits_data = load_json(config['train_file_graphs']) + \
                    load_json(config['val_file_graphs']) + \
                    load_json(config['test_file_graphs'])
    
    train_data = json.load(open(config['train_file_graphs'], 'r'))
    # train_data = []
    val_data = json.load(open(config['val_file_graphs'], 'r'))
    # val_data = []
    test_data = json.load(open(config['test_file_graphs'], 'r'))
    # test_data = []
    # punct_only_regex = re.compile(r'^[-\s]+$')
    # train_data_indices = []
    # val_data_indices = []
    # test_data_indices = []
    
    # for i, line in enumerate(train_data_og):
    #     if any([punct_only_regex.match(el.lower()) for el in line['words']]):
    #         train_data.append(line)
    #     else:
    #         train_data_indices.append(i)
    
    # for i, line in enumerate(val_data_og):
    #     if any([punct_only_regex.match(el.lower()) for el in line['words']]):
    #         val_data.append(line)
    #     else:
    #         val_data_indices.append(i)
    
    # for i, line in enumerate(test_data_og):
    #     if any([punct_only_regex.match(el.lower()) for el in line['words']]):
    #         test_data.append(line)
    #     else:
    #         test_data_indices.append(i)

    # indices = {
    #     'train': train_data_indices,
    #     'val': val_data_indices,
    #     'test': test_data_indices,
    # }

    # with open('indices_enewt.json', 'w', encoding='utf8') as f:
    #     json.dump(indices, f, ensure_ascii = False, indent = 4)

    # train_data_ratio = len(train_data) / len(train_data_og)
    # print(1 - train_data_ratio, len(train_data), len(train_data_og))
    # val_data_ratio = len(val_data) / len(val_data_og)
    # print(1 - val_data_ratio, len(val_data), len(val_data_og))
    # test_data_ratio = len(test_data) / len(test_data_og)
    # print(1 - test_data_ratio, len(test_data), len(test_data_og))
    # print(sum([len(line['words']) for line in train_data]))
    # print(sum([len(line['words']) for line in val_data]))
    # print(sum([len(line['words']) for line in test_data]))

    # allowed_indices = json.load(open('indices_enewt.json', 'r'))

    # allowed_indices_train = allowed_indices['train']
    # for i, line in enumerate(train_data_og):
    #     if i in allowed_indices_train:
    #         train_data.append(line)
    # allowed_indices_val = allowed_indices['val']
    # for i, line in enumerate(val_data_og):
    #     if i in allowed_indices_val:
    #         val_data.append(line)
    # allowed_indices_test = allowed_indices['test']
    # for i, line in enumerate(test_data_og):
    #     if i in allowed_indices_test:
    #         test_data.append(line)

    # make a single label_index_map from all the data before using it for all datasets
    config['label_index_map'] = get_mappings(all_splits_data)
    config['n_tags'] = len(config['label_index_map']['tag2class'])
    config['n_edge_labels'] = len(config['label_index_map']['edgelabel2class'])

    for loader_type in ['train', 'val', 'test']:
        if loader_type == 'train':
            train_data = GraphDataset(train_data, config, split = 'train', tokenizer = tokenizer)
            config['data_len'][loader_type] = len(train_data.data)
            if config['procedural']:
                config['dataset_max_steps'][loader_type] = train_data.max_steps
            train_loader = DataLoader(train_data, batch_size=config['batch_size'], collate_fn = collator.collate, shuffle = False)

        if loader_type == 'val':
            val_data = GraphDataset(val_data, config, split = 'val', tokenizer = tokenizer)
            config['data_len'][loader_type] = len(val_data.data)
            if config['procedural']:
                config['dataset_max_steps'][loader_type] = val_data.max_steps
            val_loader = DataLoader(val_data, batch_size=config['batch_size'], collate_fn = collator.collate)

        if loader_type == 'test':
            test_data = GraphDataset(test_data, config, split = 'test', tokenizer = tokenizer)
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