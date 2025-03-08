custom_config = {
    'train_file_graphs': './data/train.json',#['./data/val.json', './data/test.json'],
    'val_file_graphs': './data/train.json',
    'test_file_graphs': './data/test.json',
    
    'test_ignore_edge_dep': ['root', '-'], ## this will be ignored during evaluation

    'shuffle': {'train': 0, 'val': 0, 'test': 0},

    'only_use_biggest_graph': {'train': 0, 'val': 0, 'test': 0},
    'biggest_graph_threshold': 0,
    'augment_splits': {'train': 1, 'val': 0, 'test': 0},  # 1s or 0s
    'augment_k': {'train': 1, 'val': 1, 'test': 1,},
    'keep_og': 0,
    'rep_mode': 'words', # either 'words' or 'tokens'. with 'words' the bert model is no longer permutation invariant
    'use_bert_positional_embeddings': 1,
    'use_tagger_lstm': 1, # `True` is the default MTRFG setting
    'use_tag_embeddings_in_parser': 1,  # `True` is the default MTRFG setting
    'parser_lstm_hidden_size': 400,
    'use_parser_lstm': 1,  # `True` is the default MTRFG setting
    'use_gnn': 0,
    'use_step_mask': 0,
    
    'word_majority_eval': False,

    'loss_alpha': 0.5,

    'gat_conv_heads': 8,

    # 'freeze_encoder': 0,
    # 'learning_rate': 1e-4,
    'batch_size': 8,

    'save_model': False,
}

default_cfg = {
    'device': 'cuda:0', ## will be modified when we get actual device using torch
    'freeze_encoder': True,
    'freeze_tagger': False,
    'freeze_parser': False,
    'early_stopping': True,
    'patience': 30,  ## patience period for early stopping
    'epochs': 30,
    'learning_rate': 0.001,
    'shuffle' : True, 
    'save_dir': './saved_models',
    'model_name': 'bert-base-uncased', ## model name, should be key in hugging face pretrained model zoo
    'batch_size': 8,
    'encoder_output_dim': 768,
    'sparse_embedding_tags' : False,
    'n_tags': None, ## to be calculated after loading train data
    'n_edge_labels': None, ## to be calculated after loading train data
    'freeze_until_epoch' : 100, ## it freezes the encoder until a given epoch number, then unfreezes it. If freeze_until_epoch > epochs, then entire training will be with frozen encoder
    'test_ignore_tag': ['O', 'no_label'], ## this will be ignored during evaluation
    'test_ignore_edge_dep': ['root'], ## this will be ignored during evaluation
    'tag_embedding_dimension': 100,
    'seed': 27,
    'self_attention_heads': 2,
    'use_multihead_attention' : False, ## whether to use multihead attention in encoder or not! This is a learnable module that would help in generating better representations.
    'fraction_dataset': 1, ## this is between 0 and 1, if it's 0.2, then only 20% of train, test and validation dataset will be used. This is used to do hyperparameter search for training where we do training on subset of the dataset
    'betas': [0.9, 0.9],
    'use_pred_tags' : True, ## this will determine if gold tags are used for train/test/validation or not. 
    'keep_edge_labels': True,
    'softmax_scaling_coeff' : 1000,
    'gumbel_softmax' :  False, ## using gumbel softmax in tagger, it's false by default.
    'keep_tags':True, 
    'model_path' : None,
}