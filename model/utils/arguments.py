"""
	Decode passed arguments and make it in a dictionary format.
"""

import argparse
from ast import literal_eval


def build_dict_from_args(args):
	"""
		let's build a dictionary from arguments.
	"""

	assert len(args) % 2 == 0,"Override list has odd length, it must be a list of pairs"
	args_new = {}
	for key, value in zip(args[0::2], args[1::2]):
		try:
			args_new[key[2:]] = literal_eval(value)
		except:
			args_new[key[2:]] = value
	 
	return args_new


def get_args(string_args = ''):

	"""
		This will parse all the arguments and return args
	"""

	parser = argparse.ArgumentParser()

	"""
		We can consume all changes to the config through command line using this. 

		To modify any key from the config, just pass it with --<key name> as argument. 

		Remember that the key must be present in the config for it to work. Also, remember that
		if you are passing values such as list, dict or other data structures. Pass it between
		with quotes. So instead of [0.9, 0.9], it must be "[0.9, 0.9]" 
		The order must be --<key> <value> --key <value> 

		For example, if you want to change the learning rate and betas, pass them as below
			python3 train.py --opts --learning_rate "0.001" --betas "[0.9,0.9]"
	""" 
	parser.add_argument("--opts",  help="""Modify config options using the command-line 'KEY VALUE' pairs""", default=None, nargs=argparse.REMAINDER)
	args = parser.parse_args().opts
	if string_args:
		return build_dict_from_args(string_args.split())
	elif args is None:
		## return empty dictionary
		return {}
	else:
		## build a dictionary to be merged in config
		return build_dict_from_args(args)