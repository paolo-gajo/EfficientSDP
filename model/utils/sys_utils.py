from pathlib import Path
import os
import shutil
from datetime import datetime
from typing import List
from copy import deepcopy
import json
import torch
import psutil

def get_mem(show_mem: bool):
    if not show_mem:
        return None
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024**2
    return f"{mem_mb:.1f} MB"

def write_text(file_path: str, text_data: str):
    """
        write text to the file
    """
    Path(file_path).write_text(text_data)

def save_python_command(file_path, args):
    cmd = 'python3 ' + ' '.join(args) + '\n'
    write_text(file_path, cmd)

def save_reproduce_training_cmd(script_name, config, file_path):
    """
        Build full training command to get exact same training
    """
    cmd = f'python3 {script_name} --opts ' + ' '.join([f'--{key} {value}' for key, value in config.items()]) + '\n'
    write_text(file_path, cmd)

def dict_as_readable_string(input_dict):
    """
        Print dictionary in nice format!
    """
    readable_dict_string = '\n'.join([f'{key} : {value}' for key, value in input_dict.items()])
    return f'{readable_dict_string}'

def read_text(file_path: str, delimiter: str = '\n', ) -> List[str]: 
    """
        Read textfile and split with a given delimiter
    """
    return Path(file_path).read_text(encoding='utf-8').split(delimiter)

def make_dir(dirname: str):
    """
        make directory
    """
    Path(dirname).mkdir(parents=True, exist_ok=True)

def is_dir(filename: str):
    """
        check if it's a directory
    """
    return Path(filename).is_dir()

def is_file(filename: str) -> bool:
    """
        Check if path is a file
    """
    return Path(filename).is_file()

def get_name(filename: str):
    return Path(filename).stem

def get_extension(filename: str):
    """
        get extension of the file
    """
    if not is_file(filename):
        return ""
    else:
        return Path(filename).suffix

def get_parent_path(filename):
    """
        Get parent directory
    """
    return str(Path(filename).parent)

def exists(filename):
    """
        check if file exists
    """
    return Path(filename).exists()

def remove(path):
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))

## get current datetime for saving purposes
def get_current_time_string():
    return datetime.now().strftime("%Y-%m-%d--%H:%M:%S")

def load_json(json_path: str):
    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data