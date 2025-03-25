from typing import Dict
import pandas as pd

# """
# 	This file contains diffrent utility functions for 
# 	play and process Yamakata'20 dataset. We assume that
# 	input data is in CoNLLU format as provided by 
# 	Donatelli et al 21
# 	https://github.com/interactive-cookbook/ara/blob/main/Alignment_Model/preprocessing/data/Samples/sample_input_parser_gold.conllu
# """

# from .sys_utils import write_text, read_text, is_file
# # from allennlp_models.structured_prediction.dataset_readers import UniversalDependenciesDatasetReader
# # from allennlp.data.instance import Instance
# from transformers import AutoTokenizer
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# from torch.utils.data._utils.collate import default_collate
# from tqdm import tqdm
# import torch
# from typing import Dict, Tuple, List
# import string
# import networkx as nx
# # from conllu.models import TokenList
# # from conllu import parse
# import random

# def read_conllu(file: str) -> List[TokenList]:
# 	sentences = []
# 	with open(file, 'r') as f:
# 		sentences = parse(f.read())
# 	return sentences

# def _tokenlist_to_graph(tokenlist: TokenList) -> nx.DiGraph:
# 	edges = []
# 	for token in tokenlist:
# 		edges.append((token['head'], token['id']))
# 	return nx.DiGraph(edges)

# def is_arborescence(conllu_filename: str) -> bool:
# 	"""
# 		Takes a conllu file and checks if all recipes inside of it are rooted trees.
# 		If it detects non-trees, it will print some information to console to help with debugging.
# 	"""
# 	result = True
# 	data = read_conllu(conllu_filename)
# 	for i, tokenlist in enumerate(data):
# 		graph =_tokenlist_to_graph(tokenlist)
# 		if not nx.is_arborescence(graph):
# 			print('Recipe {}, starting with {} is not an arborescence'.format(i,
# 				[token['form'] for token in tokenlist[:5]])
# 			)
# 			print('One sample cycle to help you debug is:')
# 			print(next(nx.simple_cycles(graph)))
# 			print('')
# 			result = False
# 	return result

def build_augmented_recipe(recipe):
	"""
		Here we augment recipe with the step IDs. We assume that each
		step ends with ".", "?" or "!". 
		Example:
			Input: "Pour soda. Stir the mixture" will basically become
			["01_Pour", "01_soda", "01_.", "02_Stir", "02_the", "02_mixture"]

		Input: Note that the input here is CoNLLU format recipe as a list, where 
		each element would correspond to a single token!
	"""
	end_puncs = ".?!" ## this could be made better. 
	step_id = 1
	augmented_tokens, recipes_as_list_of_steps = [], ['']
	for recipe_token in recipe:
		token = recipe_token.split('\t')[1]
		augmented_tokens.append(f'{step_id}'.zfill(2) + f'_{token}')
		recipes_as_list_of_steps[-1] += f'{token} '
		if token in end_puncs:	
			step_id = step_id +  1
			recipes_as_list_of_steps[-1] = recipes_as_list_of_steps[-1].replace(f'{token} ', token)
			recipes_as_list_of_steps.append('')

	return augmented_tokens, recipes_as_list_of_steps

# def get_allennlp_in_conllu_format(datapoint: Instance):
# 	"""
# 		This takes allennlp Instance as input and get it in CoNLL-U
# 		format.
# 	"""

# 	tags = datapoint['pos_tags'].labels
# 	edge_labels = datapoint['head_tags'].labels
# 	words = [token.text for token in datapoint['words'].tokens]
# 	head_indices = datapoint['head_indices'].labels
# 	output_conllu = '\n'.join( [f'{i+1}\t{word}\t_\t_\t{tag}\t_\t{head_index}\t{edge_label}\t_\t_' for i, (word, tag, head_index, edge_label) in enumerate(zip(words, tags, head_indices, edge_labels))] )

# 	return output_conllu

# def extract_recipe_text(filepath: str) -> List[str]:
# 	"""
# 		Extract recipe text of all the recipes in CoNLLU format. 
# 		We expect that the recipes are in CoNLLU format.
# 	"""
# 	all_recipes =  read_text(filepath, delimiter = '\n\n')
# 	all_recipe_text = []

# 	for recipe in all_recipes:
# 		token_list = ' '.join([token_line.split('\t')[1] for token_line in recipe.split('\n') if len(token_line) >  0])

# 		## the punctuations should not have space before it as it may denote
# 		## end of line or a comma and so on
# 		for punc in string.punctuation:
# 			token_list = token_list.replace(f' {punc}', f'{punc}')
# 		all_recipe_text.append(token_list)

# 	return all_recipe_text

# def read_conllu_dataset_allennlp(filepath: str, keep_edge_labels=True, keep_tags=True) -> List[Instance]:
# 	"""
# 		Read CoNLLU data file using allennlp's 
# 		Universal dataset reader, and returns list 
# 		of allennlp instances, where each instance is 
# 		1 recipe/datapoint
# 	"""
# 	reader = UniversalDependenciesDatasetReader(use_language_specific_pos = True)
# 	data = list(reader._read(filepath))

# 	if not keep_edge_labels:
# 		def remove_label(label):
# 			if label == 'root':
# 				return 'root'
# 			else:
# 				return 'o'
# 		for datum in data:
# 			datum.fields['head_tags'].labels = list(map(remove_label, datum.fields['head_tags'].labels))

# 	if not keep_tags:
# 		def remove_tag(tag):	
# 			return 'O'
# 		for datum in data:
# 			datum.fields['pos_tags'].labels = list(map(remove_tag, datum.fields['pos_tags'].labels))

# 	return data

# def get_recipe_stats_from_file(filepath):
# 	"""
# 		get recipe stats from a conllu file. 
# 		We return a dict with min, max, mean and std. 
# 	"""
# 	data = read_conllu_dataset_allennlp(filepath)
# 	recipe_len = [len(datapoint['pos_tags'].labels) for datapoint in data]

# 	return {'num_recipes': len(recipe_len), 'max_len': round(np.max(recipe_len), 3),'min_len': round(np.min(recipe_len), 3), 'avg_len': round(np.mean(recipe_len), 3),'std_dev': round(np.std(recipe_len), 3)}

# def get_root_token_stats_from_file(filepath, token = 'root'):
# 	"""
# 		Find statistics about root token to understand 
# 		low precision/high recall issue.
# 	"""
# 	data = read_conllu_dataset_allennlp(filepath)
# 	root_tokens_in_recipes = [datapoint['head_tags'].labels.count(token) for datapoint in data]
	
# 	return {'num_recipes': len(root_tokens_in_recipes), f'max_{token}_tokens': round(np.max(root_tokens_in_recipes), 3), f'min_{token}_tokens': round(np.min(root_tokens_in_recipes), 3), f'avg_{token}_tokens': round(np.mean(root_tokens_in_recipes), 3), f'std_dev_{token}_tokens': round(np.std(root_tokens_in_recipes), 3)}

# def get_token_tag_pairs(filepath: str) -> List[Dict[str, List[str]]]:
# 	"""
# 		After loading input data in CoNLLU format,
# 		here we extract all tokens and corresponding 
# 		tags.
# 	"""
# 	data = read_conllu_dataset_allennlp(filepath)
	
# 	token_tag_pair = [{'tokens' : instance.fields['words'].tokens, 'tags' : instance.fields['pos_tags'].labels} for instance in data]
	
# 	return token_tag_pair

# def get_tags_to_ids(filepath: str) -> Dict[str, int]:
# 	"""
# 		 A dictionary with tag as a key and class label as value.
# 	"""

# 	data = read_conllu_dataset_allennlp(filepath)
# 	all_labels = [tag for instance in data for tag in instance.fields['pos_tags'].labels] 
# 	unique_labels = list(set(all_labels))
# 	label_count = {label: all_labels.count(label)  for label in unique_labels}
# 	label_sort = dict(sorted(label_count.items(), key=lambda item: item[1], reverse = True))
# 	tag2class = {label: i+1 for i, label in enumerate(label_sort)}
# 	tag2class.update({'no_label': 0})
	
# 	return tag2class

# def get_ids_to_tags(filepath: str) -> Dict[int, str]:
# 	"""
# 		Dictionary mapping from class label to tag
# 	"""

# 	tag2class = get_tags_to_ids(filepath)
# 	class2tag = {tag2class[label]: label  for label in tag2class}

# 	return class2tag
	
# def get_edgelabel_to_ids(filepath: str) -> Dict[str, int]:
# 	"""
# 		 A dictionary with edge label as a key and class label as value.
# 	"""

# 	data = read_conllu_dataset_allennlp(filepath)
# 	all_labels = [tag for instance in data for tag in instance.fields['head_tags'].labels]
# 	unique_labels = list(set(all_labels))
# 	label_count = {label: all_labels.count(label)  for label in unique_labels}
# 	label_sort = dict(sorted(label_count.items(), key=lambda item: item[1], reverse = True))
# 	edgelabel2class = {label: i for i, label in enumerate(label_sort.keys())}
# 	# edgelabel2class.update({'no_label': 0})
# 	return edgelabel2class

# def get_ids_to_edgelabels(filepath: str) -> Dict[int, str]:
# 	"""
# 		Dictionary mapping from class label to edge label for flow graph
# 	"""

# 	edgelabel2class = get_edgelabel_to_ids(filepath)
# 	class2edgelabels = {edgelabel2class[label]: label  for label in edgelabel2class}

# 	return class2edgelabels

# def merge_conllu_files(inputfiles: List[str], op_filepath: str = None, delimiter: str = '\n\n'):
# 	"""
# 		This function will merge multiple CoNLLU 
# 		files, to build a single CoNLLU file, and 
# 		save it at a required directory
# 	"""

# 	if not op_filepath:
# 		op_filepath = '/tmp/merged.conllu'
# 		print(f"File path is not provided, will be saving at {op_filepath}.")

# 	## let's start merging
# 	all_data = []
# 	for filepath in inputfiles:
# 		assert is_file(filepath), f"File not found at {filepath}. Please provide valid input path."
# 		all_data.extend(read_text(filepath, delimiter = delimiter))

# 	## shuffle data
# 	random.shuffle(all_data)

# 	## filter text which is empty line
# 	all_data = [data_point for data_point in all_data if len(data_point) > 0]
# 	all_data_text = '\n\n'.join(all_data)

# 	write_text(op_filepath, all_data_text)
# 	return op_filepath

# def get_label_index_mapping(train_file):
# 	"""
# 		get mapping between labels and class indices for
# 		tags and edge labels
# 	"""
# 	if isinstance(train_file, List):
# 		print("Received list of files for train, merging them.")
# 		train_file = merge_conllu_files(train_file)
# 	tag2class = get_tags_to_ids(train_file)
# 	edgelabel2class = get_edgelabel_to_ids(train_file)
# 	label_index_map = {'tag2class' : tag2class, 'edgelabel2class' : edgelabel2class}

# 	return label_index_map

# def get_index_label_mapping(train_file):
# 	"""
# 		get mapping between index and its label
# 	"""
# 	if isinstance(train_file, List):
# 		print("Received list of files for train, merging them.")
# 		train_file = merge_conllu_files(train_file)
# 	class2tag = get_ids_to_tags(train_file)
# 	class2edgelabel = get_ids_to_edgelabels(train_file)
# 	index_label_map = {'class2tag': class2tag, 'class2edgelabel': class2edgelabel}

# 	return index_label_map

# """
# 	 A dataloader class which is used to load data
# """
# class NERdataset(Dataset):
# 	def __init__(self, filepath, tokenizer, label_index_map, config = None):
# 		assert config is not None, "No config provided for dataloder."
# 		self.data = read_conllu_dataset_allennlp(filepath, config['keep_edge_labels'], config['keep_tags'])
# 		self.label_index_map = label_index_map
# 		self.tokenizer = tokenizer
# 		self.tag2class = label_index_map['tag2class']
# 		self.edgelabel2class = label_index_map['edgelabel2class']
# 		self.device = config['device']
# 		self.tensor_len = 512
# 		self.fraction_dataset = config['fraction_dataset']

# 		## let's process the dataset 
# 		if self.fraction_dataset > 0 and self.fraction_dataset < 1:
# 			self.data = self.data[:int(len(self.data) * self.fraction_dataset)]
# 			self.data_processed = [self.process_datapoint(datapoint) for datapoint in tqdm(self.data, desc=f'Processing dataset from {filepath} file with fraction {self.fraction_dataset}.')]
# 		else:
# 			self.data_processed = [self.process_datapoint(datapoint) for datapoint in tqdm(self.data, desc=f'Processing full dataset from {filepath} file.')]

# 	def build_tensor(self, label_inp, tensor_len = 512, default_val = -100):
# 		"""
# 			This function makes label tensor, of same dimension as encoded input
# 			Default value is -100 as pytorch ignores everything with class value -100
# 		"""
		
# 		label_inp_padded = [default_val] * tensor_len

# 		if len(label_inp) >= tensor_len:
# 			print("Your input is longer than max length, so we are trucating it") 
# 			label_inp_padded = label_inp[0 : tensor_len]
# 		else:
# 			label_inp_padded[ 0 : len(label_inp)] = label_inp
# 		return label_inp_padded

# 	def process_datapoint(self, datapoint):
# 		"""
# 			This will process the datapoint of the dataset
# 			and get the input into a format appropriate for
# 			the model's input.
# 		"""

# 		# step 1: get the sentence and word labels 
# 		recipe_tokenized = [token.text for token in datapoint['words'].tokens]
# 		pos_tag_labels = [self.tag2class[label] for label in datapoint['pos_tags'].labels] if 'pos_tags' in datapoint else []

# 		head_tags = [self.edgelabel2class[label] for label in datapoint['head_tags'].labels] if 'head_tags' in datapoint else []
# 		head_indices = datapoint['head_indices'].labels if 'head_indices' in datapoint else []
		
# 		# step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
# 		encoded_input = self.tokenizer(recipe_tokenized, is_split_into_words = True,  padding= 'max_length', truncation=True, return_tensors='pt', max_length = self.tensor_len)
# 		# input_words = self.tokenizer.convert_ids_to_tokens(encoded_input['input_ids'])
# 		word_ids = torch.as_tensor([elem if elem is not None else -100 for elem in encoded_input.word_ids()]) ## None can't be written to torch tensors
# 		encoded_input = {key: value.view(value.shape[-1]).to(torch.device(self.device)) for key, value in encoded_input.items()}
# 		encoded_input.update({'word_ids_custom' : word_ids.to(torch.device(self.device))})

# 		# step 3: padding labels to respect length
# 		pos_tag_labels = self.build_tensor(pos_tag_labels, tensor_len = self.tensor_len, default_val = 0)
# 		head_tags = self.build_tensor(head_tags, tensor_len = self.tensor_len, default_val = 0)
# 		head_indices = self.build_tensor(head_indices, tensor_len = self.tensor_len, default_val = 0) ## default val is 0 here as head indices with 0 are not connected to any node
# 		word_mask = self.build_tensor([1] * len(recipe_tokenized), tensor_len = self.tensor_len, default_val = 0)
# 		recipe_tokenized = self.build_tensor(recipe_tokenized, tensor_len = self.tensor_len, default_val = 'default')
		

# 		# step 4: Build input dictionary
# 		item = {}
# 		item['words'] = recipe_tokenized
# 		encoded_input.update({'words_mask_custom' : torch.as_tensor(word_mask).long().to(torch.device(self.device))})
# 		item['encoded_input'] = encoded_input
# 		item['pos_tags'] = torch.as_tensor(pos_tag_labels).to(torch.device(self.device))
# 		item['head_tags'] = torch.as_tensor(head_tags).to(torch.device(self.device))
# 		item['head_indices'] = torch.as_tensor(head_indices).to(torch.device(self.device))
		
# 		return item

# 	def __len__(self):
# 		return len(self.data)

# 	def __getitem__(self, index):
# 		return self.data_processed[index]


# def ner_collate_fuction(batch):
# 	"""
# 		We want to keep list of input texts as stacked, which can't happen with
# 		original pytorch's default collate function. So we manually merge the 
# 		text, and use default collate for all torch tensors. 
# 	"""
# 	input_text_stacked = np.array([batch_item['words'] for batch_item in batch])
# 	batch_out = default_collate(batch)
# 	batch_out['words'] = input_text_stacked

# 	return batch_out

def build_dataset_dict(config: Dict = None, dataset_name: str = 'ade'):
    data_path_train = config['data_path'].format(dataset_name = dataset_name, split = 'train')
    data_path_val = config['data_path'].format(dataset_name = dataset_name, split = 'val')
    data_path_test = config['data_path'].format(dataset_name = dataset_name, split = 'test')
    entity_dict_path = config['entity_dict_path'].format(dataset_name = dataset_name)
    rel_dict_path = config['rel_dict_path'].format(dataset_name = dataset_name)

    dataset_train = pd.read_json(data_path_train)
    dataset_val = pd.read_json(data_path_val)
    dataset_test = pd.read_json(data_path_test)

    return {
        'train': dataset_train,
        'val': dataset_val,
        'test': dataset_test,
    }