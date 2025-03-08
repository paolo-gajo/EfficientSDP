# from mtrfg.utils import merge_conllu_files, make_dir
from tqdm import tqdm
import torch
import os
import numpy as np


def get_curriculum_learning_data_paths(dataset_path, until_dataset = 0, data_strategy = 'bin_by_bin'):
    """
        This function is to get dataset paths for currculum learning model. 
        Here we merge the dataset
    """

    if data_strategy == 'bin_by_bin': ## here we only get current bin
        return os.path.join(f'{dataset_path}/train_{until_dataset - 1}.conllu'), os.path.join(f'{dataset_path}/test_{until_dataset - 1}.conllu'), os.path.join(f'{dataset_path}/dev_{until_dataset - 1}.conllu')
 
    elif data_strategy == 'cumulation': ## this is when we merge all previous bins
        train_paths = [os.path.join(f'{dataset_path}/train_{i}.conllu') for i in range(until_dataset)]
        test_paths = [os.path.join(f'{dataset_path}/test_{i}.conllu') for i in range(until_dataset)]
        dev_paths = [os.path.join(f'{dataset_path}/dev_{i}.conllu') for i in range(until_dataset)]

    else:
        print("Invalid data strategy for curriculum learning!")

    ## let's build merged files
    train_path_merged_file = os.path.join(f'{dataset_path}', f'train_0-{until_dataset}.conllu')
    test_path_merged_file = os.path.join(f'{dataset_path}', f'test_0-{until_dataset}.conllu')
    dev_path_merged_file = os.path.join(f'{dataset_path}', f'dev_0-{until_dataset}.conllu')

    ## merge the files and return filepaths
    train_path_merged_file = merge_conllu_files(inputfiles=train_paths, op_filepath=train_path_merged_file, delimiter='\n\n')
    test_path_merged_file = merge_conllu_files(inputfiles=test_paths, op_filepath=test_path_merged_file, delimiter='\n\n')
    dev_path_merged_file = merge_conllu_files(inputfiles=dev_paths, op_filepath=dev_path_merged_file, delimiter='\n\n')

    return train_path_merged_file, test_path_merged_file, dev_path_merged_file

def train_epoch(model, train_loader, optimizer, epoch):
    model.train()
    model.set_mode('train')

    with tqdm(train_loader, position=0, leave = False) as pbar:
        for inp_data in pbar:
            optimizer.zero_grad()
            loss = model(inp_data)
            
            ## if loss is nan, we keep going forward
            if torch.isnan(loss).item():
                continue
            pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item()}", refresh = True)
            loss.backward()
            optimizer.step()

    return model

def validate_epoch(model, data_loader):
    """
        runs validation
    """
    model.eval()
    model.set_mode('validation')

    ## let's get total loss over the validation set
    losses = [model(inp_data).item() for inp_data in tqdm(data_loader, position=0, leave = False)]

    return round(np.mean(losses), 3)
