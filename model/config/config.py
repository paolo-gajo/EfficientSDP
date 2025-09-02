custom_config = {
    # 'train_file_graphs': './data/{dataset_name}/bio/train_punct.json',
    # 'val_file_graphs': './data/{dataset_name}/bio/val_punct.json',
    # 'test_file_graphs': './data/{dataset_name}/bio/test_punct.json',
    'train_file_graphs': './data/{dataset_name}/bio/train.json',
    'val_file_graphs': './data/{dataset_name}/bio/val.json',
    'test_file_graphs': './data/{dataset_name}/bio/test.json',
    'dataset_name': 'ade',
    'word_majority_eval': 0,
    'save_model': 1,
    'data_len': {'train': 0, 'val': 0, 'test': 0,},
    'dataset_max_steps': {'train': 0, 'val': 0, 'test': 0,},
    'gat_conv_heads': 8,
    'test_ignore_tag': [],
    'test_ignore_edges': [],
    'test_ignore_edge_dep': [], ## this will be ignored during evaluation
    'shuffle': {'train': 1, 'val': 0, 'test': 0},
    'plot': 0,
    'debug': 0,

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
    'augment_type': 'permute', # 'permute', 'random', 'hybrid'
    'adjacency_direction': 'directed', # 'directed', 'mirrored', 'undirected'
    'save_suffix': '',
    'padding': 1,
    'procedural': 0,

    # vanilla options
    'use_tagger_rnn': 1,
    'use_parser_rnn': 1, 
    'parser_rnn_hidden_size': 400,
    'parser_rnn_layers': 3,
    'tagger_rnn_layers': 1,
    'tagger_rnn_type': 'lstm', # 'gru', 'lstm'
    'parser_rnn_type': 'lstm', # 'gru', 'lstm', 'rnn', 'normlstm', 'normrnn', 'transformer'
    'tag_embedding_type': 'linear', # 'linear', 'embedding', 'none'
    'tagger_dropout': 0.3,

    # parser options
    'parser_type': 'gat', # 'simple', 'mtrfg', 'gnn', 'gcn', 'gat', 'gat_unbatched', 'dgm', 'gnn2'
    'parser_init': 'xu', # 'xu', 'norm', 'xu+norm'
    'bma_init': 'xu', # 'xu', 'norm'
    'arc_norm': 1,
    'mlp_dropout': 0.3,

    # gat
    'gnn_layers': 1,
    'rnn_dropout': 0.3,
    'gnn_dropout': 0,
    'gnn_activation': 'tanh',

    # lstmnorm
    'rnn_residual': 0,
    'use_gnn_steps': -1,
    'parser_residual': 0,
    'bias_type': 'simple',
    'biaffine_activation': None,
    'top_k': 1,
    'num_attn_heads': 1,
    'step_bilinear_attn': 0,
    'arc_pred': 'attn',
    'use_parser_gnn': 0,
    'mhabma': 0,
    'arc_representation_dim': 500,
    'tag_representation_dim': 100,

    # triaffine options
    'triaffine_arc_norm': 0,
    'triaffine_activation': None,
    
    # graphrnn
    'graph_rnn_hidden_graph': 200,
    'graph_rnn_hidden_edge': 200,
    'graph_rnn_node_layers': 3,
    'graph_rnn_edge_layers': 3,
    'graph_rnn_m': 20,

    # lgi options
    'lgi_enc_layers': 3,
    'use_fc': 1,
    'sanity_copy_gold': 0,

    # model options
    'task_type': 'nlp', # 'nlp' or 'graph'
    'model_type': 'attn', # 'attn' or 'gen' or 'graph'
    'lgi_gat_type': 'base',
    'gat_norm': 0,
    'graph_rnn_pred_type': 'bilinear', # 'simple' or 'bilinear'
    'freeze_encoder': 1,
    'use_lora': 0,
    'use_pred_tags': 1, ## this will determine if gold tags are used for train/test/validation or not 
    'model_name': 'bert-base-uncased',
    # 'model_name': 'bert-base-multilingual-cased',
    # 'model_name': 'microsoft/deberta-v3-base',
    # 'model_name': 'microsoft/deberta-v3-large',
    # 'model_name': 'google-bert/bert-large-uncased',
    'seed': 0,
    'tagger_lambda': 0.1,
    'parser_lambda': 1,
    'rep_mode': 'words', # either 'words' or 'tokens'
    'laplacian_pe': None, # 'encoder' or 'parser'
    'use_abs_step_embeddings': 0,
    'learning_rate_encoder': 1e-4,
    'learning_rate_freeze': 1e-3,
    'learning_rate_large': 3e-5,
    'use_step_mask': 0,
    'use_bert_positional_embeddings': 1,
    'unfreeze_layers': [],
    'use_encoder_attn': 0,
    'output_edge_scores': 1,
    'tag_dropout': 0.2,
    'current_step': 0,

    # training options
    'use_warmup': 0,
    'warmup_ratio': 0.06, # percentage of steps over which to warm up
    'scheduler_type': 'linear', # `linear` or `cosine`
    'use_clip_grad_norm': 0,
    'grad_clip_norm': 1.0,
    'batch_size': 8,
    'training': 'steps',
    'train_steps': 10000,
    'eval_steps': 500,
    'eval_samples': 0, # 0 = all samples
    'epochs': 0 ,
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
    'save_dir': '',
    'model_name': 'bert-base-uncased', ## model name, should be key in hugging face pretrained model zoo
    'batch_size': 8,
    'encoder_output_dim': 768,
    'sparse_embedding_tags' : False,
    'n_tags': None, ## to be calculated after loading train data
    'n_edge_labels': None, ## to be calculated after loading train data
    'freeze_until_epoch' : 999, ## it freezes the encoder until a given epoch number, then unfreezes it. If freeze_until_epoch > epochs, then entire training will be with frozen encoder
    'test_ignore_tag': ['O', 'no_label'], ## this will be ignored during evaluation
    'test_ignore_edge_dep': ['root'], ## this will be ignored during evaluation
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
