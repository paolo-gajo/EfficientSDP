"""
    Through this script, we will train a series of models for tagging and parsing, to eventually benchmark their performances
"""

import os
from stepparser.utils import make_dir

"""
    List of all the models to be trained
"""
model_names = ['bert-base-uncased', 'roberta-base', 'albert-base-v1', 'albert-large-v1', 'xlm-roberta-base', 'facebook/bart-base', 'microsoft/layoutlm-base-uncased', 'sentence-transformers/all-MiniLM-L6-v2', 'dslim/bert-base-NER', 'xlm-roberta-large-finetuned-conll03-english', 'google/electra-small-discriminator']

save_dir = "saved_models/FG_parsing_with_sparse_embedding/"
make_dir(save_dir)

models_failed = []

## let's start training!
for model_name in model_names:
    train_cmd = f'CUDA_VISIBLE_DEVICES=1 python3 tools/train.py --opts --model_name {model_name} --save_dir {save_dir} --sparse_embedding_tags "True"'
    if os.system(train_cmd) != 0:
        models_failed.append(model_name)
    
## we must know for which models our script failed
if len(models_failed) > 0:
    model_failed_str = ', '.join(models_failed)
    print(f"Training script failed for {model_failed_str}.")

## consolidate results in form of a csvs