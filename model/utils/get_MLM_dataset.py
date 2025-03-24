
"""
	This is to get Recipe1M or YouCookII dataset!
"""

import os
import string
import json

"""
	Make dictionary for different datasets.
"""
ann_url_dict = {"YouCookII": 'http://youcook2.eecs.umich.edu/static/YouCookII/youcookii_annotations_trainval.tar.gz',
			"Recipe1M": 'http://data.csail.mit.edu/im2recipe/recipe1M_layers.tar.gz'}

ann_file_names = {"YouCookII": 'youcookii_annotations_trainval.json',
				"Recipe1M": 'layer1.json'}

## add punctuation 
def add_punctuation(recipe_text):
	recipe_text = [line if line[-1] in string.punctuation else line+'.' for line in recipe_text]
	return recipe_text

## if json file is not found or not provided
## we download it from the server and store it
def download_json_annotation(file_url, data_dir):
	## download the file
	cmd = f'wget {file_url} -P {data_dir}'
	if os.system(cmd) != 0:
		print(f"Failed to download the dataset annotation file from {file_url}.")
		exit()

	## untar it
	tar_file_path = os.path.join(data_dir,os.path.basename(file_url))
	cmd = f'tar -xvf {tar_file_path} --directory {data_dir}'
	if os.system(cmd) != 0:
		print(f"Failed to extract file at {tar_file_path}.")
		exit()	

	## remove tar file
	os.system(f'rm {tar_file_path}')

## let's extract recipe text from annotation json file
def build_train_dev_list(json_data, dataset = 'YouCookII'):

	train_list = []
	dev_list = []

	if dataset == 'YouCookII':
		youcook_data = json_data['database']
		## let's go through youcook2 dataset and get all the text
		for vid_id in youcook_data:
			## if the recipe type is validation, write to validation file
			if youcook_data[vid_id]['subset'] == 'validation':
				dev_list.append(' '.join(add_punctuation([ann['sentence'] for ann in youcook_data[vid_id]['annotations']])))

			## if the recipe type is training, write to the train file
			elif youcook_data[vid_id]['subset'] == 'training':
				train_list.append(' '.join(add_punctuation([ann['sentence'] for ann in youcook_data[vid_id]['annotations']])))

	elif dataset == 'Recipe1M':

		## let's iterate through the data
		for recipe in json_data:
			if recipe['partition'] == 'train':
				train_list.append(' '.join(add_punctuation([instruction['text'] for instruction in recipe['instructions']])))
			elif recipe['partition'] == 'val':
				dev_list.append(' '.join(add_punctuation([instruction['text'] for instruction in recipe['instructions']])))
	else:
		print(f"Unsupported dataset {dataset}.")
		exit()
	train_list = add_punctuation(train_list)
	dev_list = add_punctuation(dev_list)
	return train_list, dev_list