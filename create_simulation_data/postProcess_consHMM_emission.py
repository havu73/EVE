import pandas as pd
import numpy as np
import helper
import argparse
import os, sys

def transform_emission_df(raw_emission_fn, state_reorder_fn, output_fn):
	emission_df = pd.read_csv(raw_emission_fn, header = 0, index_col = 0, sep = '\t')
	state_reorder_df = pd.read_csv(state_reorder_fn, header = None, index_col = 0, sep = '\t') # index as the first column, which shows that raw state numbers
	state_reorder_df.columns = ['reordered_state']
	emission_df = emission_df.merge(state_reorder_df, left_index = True, right_index = True)
	emission_df.sort_values(by = ['reordered_state'], inplace = True)
	emission_df.rename(columns ={'reordered_state': 'state'}, inplace = True) 
	emission_df.to_csv(output_fn, header = True, index = False, sep = '\t')
	return 

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'raw ConsHMM emission matrix provided online is actually before the preprocessing step, we want to rearrage the states so that they follow the same order as in the publication')
	parser.add_argument('--raw_emission_fn', type = str, required = False, default = '/u/home/h/havu73/project-ernst/program_source/ConsHMM/models/hg19_multiz100way/emissions_100.txt', help = 'raw_emission_fn, outputted from ChromHMM')
	parser.add_argument('--state_reorder_fn', type = str, required = False, default = '/u/home/h/havu73/project-ernst/program_source/ConsHMM/stateRenamingFiles/hg19_multiz100way_state_renaming_100.tsv', help= 'Mapping of raw states to published states, raw from ChromHMM, published correspond to those. presented in the paper')
	parser.add_argument('--output_fn', type = str, required = False, default = '/u/home/h/havu73/project-ernst/ncEVE_project/training_data/simulation/consHMM_emission.txt')
	args = parser.parse_args()
	print(args)
	helper.check_file_exist(args.raw_emission_fn)
	helper.check_file_exist(args.state_reorder_fn)
	helper.create_folder_for_file(args.output_fn)
	print('Done getting command line argument')
	transform_emission_df(args.raw_emission_fn, args.state_reorder_fn, args.output_fn)
	print('Done! :)')