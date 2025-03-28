custom_config = {
    'train_file_graphs': './data/{dataset_name}/train.json',
    'val_file_graphs': './data/{dataset_name}/val.json',
    'test_file_graphs': './data/{dataset_name}/test.json',
    'dataset_name': 'yamakata',
    'word_majority_eval': 0,
    'save_model': 1,
    'data_len': {'train': 0, 'val': 0, 'test': 0,},
    'dataset_max_steps': {'train': 0, 'val': 0, 'test': 0,},
    'parser_lstm_hidden_size': 400,
    'gat_conv_heads': 8,
    'test_ignore_edge_dep': ['root', '-'], ## this will be ignored during evaluation
    'shuffle': {'train': 1, 'val': 0, 'test': 0},
    'tagger_lambda': 0.1,
    'parser_lambda': 1,
    'batch_size': 4,
    'plot': 0,

    # data options
    'only_use_biggest_graph': {'train': 0, 'val': 0, 'test': 0},
    'biggest_graph_threshold': 0,
    'augment_train': 0,
    'augment_val': 0,
    'augment_test': 0,
    'augment_k_train': 0,
    'augment_k_val': 0,
    'augment_k_test': 0,
    'keep_og_train': 1,
    'keep_og_val': 1,
    'keep_og_test': 1,
    'augment_type': 'random', # 'permute', 'random', 'hybrid'
    'adjacency_direction': 'directed', # 'directed', 'mirrored', 'undirected'
    'results_suffix': '_steps',
    'padding': 1,
    'procedural': 0,

    # tagger options
    'use_tagger_lstm': 0,

    # parser options
    'step_bilinear_attn': 0,
    'arc_pred': 'attn',
    'use_parser_gnn': 0,
    'use_parser_lstm': 0,
    'use_tag_embeddings_in_parser': 1,
    'mhabma': 0,
    'parser_type': 'mtrfg', # 'mtrfg' or 'gnn'
    'gnn_enc_layers': 1,

    # model options
    'seed': 27,
    'rep_mode': 'words', # either 'words' or 'tokens'
    'laplacian_pe': '', # 'encoder' or 'parser'
    'use_abs_step_embeddings': 0,
    'freeze_encoder': 1,
    'learning_rate': 1e-3,
    'use_gnn': '0',  # 'gat' or 'mpnn'
    'use_step_mask': 0,
    'use_bert_positional_embeddings': 1,
    # 'one_hot_tags': 0, # TODO: maybe 0 needs lstm
    

    # training options
    'training': 'steps',
    'training_steps': 2000,
    'eval_steps': 100,
    'epochs': 100,
    'patience': 0.3,
}

default_cfg = {
    'device': 'cuda:0', ## will be modified when we get actual device using torch
    'freeze_encoder': True,
    'freeze_tagger': False,
    'freeze_parser': False,
    'early_stopping': True,
    'patience': 30,  ## patience period for early stopping
    'epochs': 100,
    'learning_rate': 0.001,
    'shuffle' : True, 
    'save_dir': './results',
    'model_name': 'bert-base-uncased', ## model name, should be key in hugging face pretrained model zoo
    'batch_size': 8,
    'encoder_output_dim': 768,
    'sparse_embedding_tags' : False,
    'n_tags': None, ## to be calculated after loading train data
    'n_edge_labels': None, ## to be calculated after loading train data
    'freeze_until_epoch' : 999, ## it freezes the encoder until a given epoch number, then unfreezes it. If freeze_until_epoch > epochs, then entire training will be with frozen encoder
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
    'use_mst_decoding_for_validation': 1,
}