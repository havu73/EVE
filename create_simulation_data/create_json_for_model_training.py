import json
import pandas as pd 
import os
import numpy as np
import argparse
import helper
from itertools import product
def get_json_value_from_param_path(json_obj, param_path):
	'''
	Reference: https://stackoverflow.com/questions/21028979/how-to-recursively-find-specific-key-in-nested-json
	'''
	if type(param_path) == str:
		param_path = param_path.split(".")

	if type(param_path) != list or len(param_path) == 0:
		return

	key = param_path.pop(0)

	if len(param_path) == 0:
		try:
			return json_obj[key]
		except KeyError:
			return

	if len(param_path):
		return get_json_value_from_param_path(json_obj[key], param_path)

def change_json_value_from_path(json_obj: dict, param_path: list, value: str): 
	'''
	Reference: https://stackoverflow.com/questions/67148983/python-change-value-in-json-object-based-on-list-of-keys
	'''
	tmp_dict = json_obj
	for p in param_path[:-1]:
		tmp_dict = tmp_dict[p]
	tmp_dict[param_path[-1]] = value
	return json_obj

def list_all_paths_within_json(json_input):
	if isinstance(json_input, dict) and len(json_input) > 1:
		result = []
		for k in json_input.keys():
			k_list = list(product([k], list_all_paths_within_json(json_input[k])))
			print(k_list)
			result.append(k_list)
		return  result
	elif isinstance(json_input, dict) and len(json_input) == 1 and isinstance(list(json_input.values())[0], list):
		return list(json_input.keys())

def product_of_multiple_lists(list_list):
	if len(list_list)  == 1:
		return list_list
	else:
		return list(product(list_list[0], product_of_multiple_lists(list_list[1:])))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'based on a json of default value, and another json of parameter search spaces, create a bunch of json files in separate folders so that we can do model training later')
	parser.add_argument('--def_json_fn', type = str, required = True, help = 'json of default values')
	parser.add_argument('--search_json_fn', type =  str, required = True, help = 'json of search values')
	parser.add_argument('--output_folder', type = str, required = True, help = 'output_folder')
	args = parser.parse_args()
	helper.check_file_exist(args.def_json_fn)
	helper.check_file_exist(args.search_json_fn)
	helper.make_dir(args.output_folder)
	print('Done getting command line arguments')
	def_js = json.load(open(args.def_json_fn))
	search_js = json.load(open(args.search_json_fn))
	search_js_path_list  = list_all_paths_within_json(search_js)
	get_json_value_from_param_path(search_js, list(search_js_path_list[0][0]))
	search_js_value_list = list(map(lambda x: get_json_value_from_param_path(search_js, x), search_js_path_list)) # list of list of possible values  for each  parameter
	search_js_value_list = product_of_multiple_lists(search_js_value_list)
	t = list_all_paths_within_json(def_js)
	print(list(t))