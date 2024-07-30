import pandas as pd
import numpy as np
import helper
import argparse
import os, sys
def get_ALPHABET_DICT():
	ALPHABET = 'ACTGXN'
	I_LIST_FOR_ALPHABET = list(range(4)) # ACTG only
	I_TO_CHAR_DICT = {}
	CHAR_TO_I_DICT = {}
	I_TO_NOCHAR_LIST_DICT = {}
	for i, char in enumerate(ALPHABET):
		I_TO_CHAR_DICT[i] = char
		CHAR_TO_I_DICT[char] = i
		I_TO_NOCHAR_LIST_DICT[i] = list(I_LIST_FOR_ALPHABET[:i] + I_LIST_FOR_ALPHABET[i+1:4])
	return ALPHABET, I_TO_CHAR_DICT, CHAR_TO_I_DICT, I_TO_NOCHAR_LIST_DICT

ALPHABET, I_TO_CHAR_DICT, CHAR_TO_I_DICT, I_TO_NOCHAR_LIST_DICT = get_ALPHABET_DICT()
print(I_TO_NOCHAR_LIST_DICT)

def read_emission_fn(emission_fn, state_list):
	emission_df = pd.read_csv(emission_fn, header = 0, index_col = None, sep = '\t')
	emission_df.rename(columns = {'state (Emission order)' : 'state'}, inplace = True)
	emission_df = emission_df[emission_df.state.isin(state_list)]
	emission_df.set_index('state', inplace = True)
	align_df_colname_list = list(emission_df.columns[emission_df.columns.str.endswith('_aligned')])
	align_df = emission_df[align_df_colname_list]
	species_list = list(map(lambda x: x.split('_aligned')[0], align_df_colname_list))
	match_df_colname_list = list(map(lambda x: '{}_matched'.format(x), species_list))
	match_df = emission_df[match_df_colname_list]
	align_df.columns = species_list
	match_df.columns = species_list
	return match_df, align_df, species_list

def sample_nonAlign_per_row(align_prob):
	align_result = np.random.binomial(1, align_prob, len(align_prob))
	align_result[align_result==0] = CHAR_TO_I_DICT['X'] #  responding to X
	return align_result


def sample_match_seq_per_row(row, match_prob, species_list, refSpec_name):
	match_result = np.random.binomial(1, match_prob, len(match_prob))
	match_result = pd.Series(match_result, index = match_prob.index)
	match_species = (row[species_list][(row[species_list] != CHAR_TO_I_DICT['X']) & (match_result == 1)]).index
	row[match_species]  = row[refSpec_name]
	NONmatch_species = (row[species_list][(row[species_list] != CHAR_TO_I_DICT['X']) & (match_result != 1)]).index
	NONmatch_bp_list = I_TO_NOCHAR_LIST_DICT[row[refSpec_name]]
	row[NONmatch_species] =  np.random.choice(NONmatch_bp_list, len(NONmatch_species), replace = True)
	return row

def sample_multiSpecies_align(align_df, match_df, species_list, state, num_bp_per_state, refSpec_name, output_columns):
	align_prob = align_df.loc[state]
	match_prob = match_df.loc[state]
	result_df = pd.DataFrame(map(lambda	x: sample_nonAlign_per_row(align_prob), range(num_bp_per_state)))
	result_df.columns = align_df.columns
	result_df[refSpec_name] = np.random.choice(np.arange(4), num_bp_per_state, replace = True)
	result_df = result_df.apply(lambda x: sample_match_seq_per_row(x, match_prob, species_list, refSpec_name), axis = 1) # apply function to each row
	result_df = result_df.applymap(lambda x: I_TO_CHAR_DICT[x]) # apply function to every value in the dataframe
	result_df['chrom'] = 'chr' # this is just so that we can obtain a file that looks like required input to the EVE model
	result_df['chosen_bp'] = state # this is just so that we can obtain a file that looks like required input to the EVE model
	result_df['state'] = 'U{}'.format(state)
	result_df = result_df[output_columns]# rearrange columns to the desired order, which is very important for putting it back into the ncEVE model
	print('Done simulating data for state {}'.format(state))
	return result_df

def get_desired_columns(sample_output_fn):
	sample_df = pd.read_csv(sample_output_fn, header = 0, index_col = None, sep = '\t', nrows = 1)
	return sample_df.columns

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Create a list of multispecies alignment data based on consHMM emission parameters of sequence matching and align with the human genome')
	parser.add_argument('--emission_fn', type = str, required = True, help = 'path to consHMM emission parameters')
	parser.add_argument('--output_fn', type = str, required = True, help = 'output_fn, following the same format as the input file to teh ncEVE model')
	parser.add_argument('--state_list', type = int, nargs='+', required = False, default = [1, 2, 50, 74, 97], help = 'List of consHMM state (format: 1,2,...) that we will use the consHMM emission parameters to sample the multispecies sequence alignment data') # multiple integer values as input https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
	parser.add_argument('--num_bp_per_state', type = int, required = False, default = 10000, help = 'number of bp to sample per consHMM state')
	parser.add_argument('--refSpec_name', type = str, required = False, default = 'hg19', help = 'Name of the reference specieces for the multispecies sequence alignment')
	parser.add_argument('--sample_output_fn', type = str, required = False, default = '~/project-ernst/ncEVE_project/training_data/onePercent/sampled_msa_gw.txt.gz', help = 'Fn so the sample output_fn, so that we know the correct order of  columns for the model')
	parser.add_argument('--seed', type=int, default=731995, help='Random seed')
	args = parser.parse_args()
	command = helper.print_command(sys.argv[0], args)
	print(args)
	helper.check_file_exist(args.emission_fn)
	helper.create_folder_for_file(args.output_fn)
	np.random.seed(args.seed)
	print('Done getting input arguments!')
	output_columns = get_desired_columns(args.sample_output_fn)
	match_df, align_df, species_list = read_emission_fn(args.emission_fn, args.state_list)
	print(match_df)
	print(align_df)
	result_df_list = []
	for state in args.state_list:
		result_df = sample_multiSpecies_align(align_df, match_df, species_list, state, args.num_bp_per_state, args.refSpec_name, output_columns)
		result_df_list.append(result_df)
	result_df = pd.concat(result_df_list)
	helper.save_gzip_dataframe_with_comment(result_df, command, args.output_fn)
	print('Done! :)')