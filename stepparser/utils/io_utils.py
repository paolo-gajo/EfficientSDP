"""
    utilities relevant to saving and loading files with popular extensions
    such as json, csv, txt etc etc
"""

import pickle
import csv
import simplejson as json
from collections import OrderedDict


### pretty_print_json
def pretty_print_json(obj, string = '', indent = 4):
    print(f'{string}{json.dumps(obj, indent=indent, sort_keys=True)}')

### save object in json file
def save_json(obj, filename, indent=4):
    with open(filename, "w") as fp:
        json.dump(obj, fp, indent=indent)

### load object from json file
def load_json(filename):
    if isinstance(filename, str):
        with open(filename, "r") as fp:
            obj = json.load(fp, object_pairs_hook = OrderedDict)
    if isinstance(filename, list):
        obj = []
        for el in filename:
            with open(el, "r") as fp:
                obj += json.load(fp, object_pairs_hook = OrderedDict)
    return obj

### Save python object as dictionary
def save_object_to_json(obj, filename):
    save_json(obj.__dict__, filename)
    
### Load python object from dictionary
def load_object_attributes_from_json(filename, obj):
    params = load_json(filename)
    for param in params:
        obj.__dict__[param] = params[param]
    return obj