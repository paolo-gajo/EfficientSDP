import warnings
from tqdm import tqdm
import torch
import time
import numpy as np
import spacy
import re
from .sys_utils import write_text
from model.utils.data_utils import reorder_tensor
from model.evaluation.evaluator import evaluate_model_nlp, evaluate_model_graph
nlp = spacy.load('en_core_web_md')
nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab, suffix_search = re.compile(r'''\.|\,|\;|\(|\)|\$''').search)

EVAL_LIST = {
    'nlp': evaluate_model_nlp,
    'graph': evaluate_model_graph,
}

def run_evaluation(model, data_loader, config = None, label_index_map = None, epoch = None, steps = None):
    assert config is not None, "No config file to run evaluation."
    
    model.eval()
    model.set_mode('test') ## this tells the model that we are testing it, so it will return us precision instead of loss

    eval_function = EVAL_LIST[config['task_type']]

    if config['task_type'] == 'nlp':
        results = eval_function(model,
                                data_loader,
                                label_index_map,
                                ignore_tags = config['test_ignore_tag'],
                                ignore_edges = config['test_ignore_edges'],
                                ignore_edge_labels = config['test_ignore_edge_dep'],
                                )
    elif config['task_type'] == 'graph':
        results = eval_function(model,
                                data_loader,
                                )
    times = results.pop('times')
    mean_inf_time, std_dev = round(np.mean(times), 3), round(np.std(times), 3)
    model_summary = get_model_summary_in_dict(model)
    model_summary['inference time'] = f'{mean_inf_time} +/- {std_dev}'

    results['epoch'] = epoch
    results['steps'] = steps
    model_summary['model_name'] = config['model_name']

    return results, model_summary

def consolidate_all_runs(all_data):
    """
        get mean, std, min and max from the list and printable format!
    """
    return round(np.mean(all_data), 4), round(np.std(all_data), 4), round(np.max(all_data), 4), round(np.min(all_data), 4) 

def get_overall_results(results_list):
    """
        This function will yield results and put it in a string to be printed.
        results_list: List of dictionaries of results!
    """

    keys = results_list[0].keys()

    results_string = []

    ## for each kind of result, we will get P,R,F1
    for key in keys:
        
        ## gather all P, R, F1 values
        all_P = [result[key]['P'] for result in results_list]
        all_R = [result[key]['R'] for result in results_list]
        all_F1 = [result[key]['F1'] for result in results_list]

        ## get mean, variance, max and min
        P_mean, P_std, P_max, P_min = consolidate_all_runs(all_P)
        R_mean, R_std, R_max, R_min = consolidate_all_runs(all_R)
        F1_mean, F1_std, F1_max, F1_min = consolidate_all_runs(all_F1)

        ## properly printing them all!
        print_start = f'{key}: '
        print_start_spaces = ' ' * len(print_start)
        p_or_m = u"\u00B1"

        ## building a results str for a single value
        result_str = f'{print_start}Precision: {P_mean} {p_or_m} {P_std}, Max precision: {P_max}, Min precision: {P_min}\n'
        result_str += f'{print_start_spaces}Recall: {R_mean} {p_or_m} {R_std}, Max Recall: {R_max}, Min Recall: {R_min}\n'
        result_str += f'{print_start_spaces}F1: {F1_mean} {p_or_m} {F1_std}, Max F1: {F1_max}, Min F1: {F1_min}\n'

        ## appending results string
        results_string.append(result_str)
    
    return '\n'.join(results_string)

def get_output_in_conllu_format(outputs, index_label_map):
    """
        Get output in CoNLL-U format from predicted outputs
    """ 
    output_conllu = []

    for recipe in outputs:
        tags = [index_label_map['class2tag'][tag_id] for tag_id in recipe['pos_tags_pred']]
        edge_labels = [index_label_map['class2edgelabel'][edge_id] for edge_id in recipe['head_tags_pred']]
        words = recipe['words']
        head_indices = recipe['head_indices_pred']
        output_conllu.append('\n'.join( [f'{i+1}\t{word}\t_\t_\t{tag}\t_\t{head_index}\t{edge_label}\t_\t_' for i, (word, tag, head_index, edge_label) in enumerate(zip(words, tags, head_indices, edge_labels))] ))

    return '\n\n'.join(output_conllu) + '\n\n'

def build_conllu_file_from_recipe_list(list_of_recipes, file_path):
    """
        This function will write a list of recipes to a 
        conllu file, which can be used for inference on 
        new recipes
    """
    
    all_recipes = []
    for recipe in list_of_recipes:
        all_recipes.append('\n'.join([f'{i+1}\t{token.text}\t_\t_\tO\t_\t0\troot\t_\t_'  for i, token in enumerate(nlp(recipe))]))
     
    all_recipes = '\n\n'.join(all_recipes) + '\n\n'
    
    write_text(file_path, all_recipes)
    

def get_model_summary_in_dict(model):
    """
        Through this function, we will summarize the 
        model
    """
    
    ## memory
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = round((mem_params + mem_bufs) / 10**6, 3) # in MB

    ## number of learnable params
    total_model_params = sum(p.numel() for p in model.parameters())
    learnable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    if hasattr(model, 'tagger'):
        tagger_params = sum(p.numel() for p in model.tagger.parameters())
    else:
        tagger_params = 0
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    parser_params = sum(p.numel() for p in model.parser.parameters())

    mem_dict = {
                'memory in (MB)' : mem,
                'total params' : total_model_params,
                'learnable params' : learnable_params,
                'total encoder params' : encoder_params,
                'total tagger params' : tagger_params,
                'total parser params' : parser_params
                }

    return mem_dict


def get_index_label_map(label_index_map):
    """
        Get index to label map, from label to index map!
    """

    class2tag = {value: key for key, value in label_index_map['tag2class'].items()}
    class2edgelabel = {value: key for key, value in label_index_map['edgelabel2class'].items()}

    return {'class2tag': class2tag, 'class2edgelabel' :class2edgelabel}


def run_inference(model, data_loader, config = None, label_index_map = None):
    
    """
        This is used to get the output during inference! 
        Here, we will get the output and return them as a list of recipes,
        which can be written to any output file as needed! 
    """

    assert config is not None, "No config file to run inference"
    assert label_index_map is not None, "No label index map to run inference!"

    
    model.eval()
    model.set_mode('test') ## this tells the model that we are testing it, so it will return us precision instead of loss
    outputs = []
    inputs = []

    ## let's time it and find average time
    times = []
    with torch.no_grad():
        with tqdm(data_loader, position=0, leave = False) as pbar:
            for inp_data in tqdm(data_loader, position=0, leave = False):
                inputs.extend(inp_data)
                st_time = time.time()
                outputs.extend(model(inp_data))
                tot_time = round((time.time() - st_time) / data_loader.batch_size, 3)
                pbar.set_description(f"Batch inference time is {tot_time} seconds", refresh = True)
                times.append(tot_time)

    mean_inf_time, std_dev = round(np.mean(times), 3), round(np.std(times), 3)

    index_label_map = get_index_label_map(label_index_map)

    outputs = get_output_in_conllu_format(outputs, index_label_map)

    return outputs

def check_perm_eq(h: torch.Tensor, indices: torch.Tensor):
    B = h.shape[0]
    for i in range(B):
        for j in range(B):
            if i != j:
                t1 = reorder_tensor(h[i], indices[i], permutation=torch.arange(1, torch.max(indices[i])))
                t2 = reorder_tensor(h[j], indices[j], permutation=torch.arange(1, torch.max(indices[j])))
                print(torch.allclose(t1, t2, rtol=0.0001, atol=0.0001))