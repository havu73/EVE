import pandas as pd 
import numpy as np 
import os 
import argparse
import helper
parser = argparse.ArgumentParser(description="Find positive pairs of variants")
parser.add_argument("--msa_folder", type=str, required=True, help="Folder where each chrom's MSA data is store")
parser.add_argument('--msa_file_suffix', type = str, required = False, default = 'maf_sequence.csv.gz', help='Each file of msa for a chrom should have the form chr\{chrom\}_\{msa_file_suffix\}')
parser.add_argument("--consHMM_folder", type=str, required=True, help="The location of consHMM segmentation")
parser.add_argument("--output_folder", type = str, required=True, help = 'Where we save the data of multi_species sequence alignment that we sample for training. Garbage in garbage out, so we are on a mission to make the best most well balanced training data here.')
parser.add_argument('--window_around_selected_bp', type = int, required = False, default = 0, help ='If this number >0, then for each bp that we select for training, we will also add the N bp left and N bp right of the selected variant to the output data (used for training later). N = window_around_selected_bp.')
parser.add_argument('--seed', type = int, required = False, default = 731995, help = 'seed for random number generator')
parser.add_argument('--num_bp_per_state', type = int, required = False, default = 300000, help = 'We sample the genome such that each state is equally present in the sampled region, so this parameter is helpful. Default value comes down to 1 percent of the genome.')
parser.add_argument('--num_consHMM_state', type = int, required = False, default = 100, help = 'num_consHMM_state')
parser.add_argument('--chrom_length_fn', type = str, required = True, help='Two columns (chrom and length), no headers, tab seperated')

def find_num_bp_per_state_per_chrom(chrom_length_fn, num_bp_per_state):
	'''
	Given the genome-wide, users want to sample num_bp_per_state bps per consHMM state, we would calculate the num_bp_per_stae_per_chrom sampled for each state in each chrom.
	Output: df with chrom, num_bp_per_state_per_chrom
	'''
	chrom_df = pd.read_csv(chrom_length_fn, header = None, index_col = None, sep = '\t')
	chrom_df.columns = ['chrom', 'length']
	chrom_list = list(map(lambda x: 'chr{}'.format(x), helper.CHROMOSOME_LIST))
	chrom_df = chrom_df[chrom_df['chrom'].isin(chrom_list)]
	chrom_df['frac_in_gene']= chrom_df['length'] / np.sum(chrom_df['length'])
	chrom_df['num_bp_per_state'] = np.ceil(chrom_df['frac_in_gene'] * num_bp_per_state).astype(int)
	chrom_df.index = chrom_df['chrom']
	return chrom_df

def sample_bp_within_consHMM_segment(row):
	return np.random.choice(np.arange(row['start'], row['end']), 1)[0]

def sample_from_consHMM_per_chrom(consHMM_fn, num_bp_per_state, num_consHMM_state, save_fn):
	'''
	We will assume that consHMM_fn will only show the segmentation for one chromosome only, not genome_wide
	'''
	df = pd.read_csv(consHMM_fn, header = 0, index_col = None, sep = '\t')
	df.columns = ['chrom', 'start', 'end', 'state']
	df = df.groupby('state')
	result_df_list = []
	for state, state_df in df:
		state_df['state'] = state
		try:
			chosen_indices = np.random.choice(state_df.index, num_bp_per_state, replace = False)
		except:
			chosen_indices = list(state_df.index)
		state_df = state_df.loc[chosen_indices, :]
		result_df_list.append(state_df)
	result_df = pd.concat(result_df_list, ignore_index = True)
	# now, we will sample 1 bp within each line
	result_df['chosen_bp'] = result_df.apply(sample_bp_within_consHMM_segment, axis = 1)
	result_df = result_df.sort_values('chosen_bp', ascending = True)
	result_df = result_df[['chrom', 'chosen_bp', 'state']]
	result_df.to_csv(save_fn, header = True, index = False, sep = '\t', compression = 'gzip')
	return result_df


def select_sampled_position_with_MSA_data_one_chrom(msa_folder, msa_file_suffix, consHMM_df, window_around_selected_bp, chrom, sampled_msa_save_fn):
	'''
	chrom is assumed to be of the form '1' and not 'chr1'
	consHMM_df only contains data for one chromosome
	'''
	msa_fn = os.path.join(msa_folder, 'chr{c}_{s}'.format(c = chrom, s = msa_file_suffix))
	msa_df = pd.read_csv(msa_fn, header = 0, index_col = 0, sep = ',')
	msa_df = msa_df.loc[consHMM_df['chosen_bp'], :]
	msa_df = pd.concat([msa_df, consHMM_df], axis = 1, ignore_index = True)
	msa_df.to_csv(sampled_msa_save_fn, header = True, index = False, sep = '\t', compression = 'gzip')
	return 

if __name__ == '__main__':
	args = parser.parse_args()
	np.random.seed(args.seed)
	helper.check_dir_exist(args.msa_folder)
	helper.check_dir_exist(args.consHMM_folder)
	helper.make_dir(args.output_folder)
	# 0. number of bp sampled in each chrom for each state
	chrom_sample_size_df = find_num_bp_per_state_per_chrom(args.chrom_length_fn, args.num_bp_per_state) # chrom, length, frac_in_gene, num_bp_per_state. index of the df: chrom
	# 1. sample positions such that each consHMM state is equally present
	for chrom in helper.CHROMOSOME_LIST:
		num_bp_per_state_this_chrom = chrom_sample_size_df.loc['chr{}'.format(chrom), 'num_bp_per_state']
		save_fn = os.path.join(args.output_folder, 'chr{}_sampled_bp.txt.gz'.format(chrom))
		if os.path.isfile(save_fn):
			consHMM_df = pd.read_csv(save_fn, header = 0, index_col = None, sep = '\t')
		else: 
			consHMM_fn = os.path.join(args.consHMM_folder, 'chr{}/chr{}_segmentation.bed.gz'.format(chrom, chrom))
			consHMM_df = sample_from_consHMM_per_chrom(consHMM_fn, num_bp_per_state_this_chrom, args.num_consHMM_state, save_fn)
		# 2. intersect sampled position with the MSA data
		# sampled_msa_save_fn  = os.path.join(args.output_folder, 'chr{}_sampled_msa.txt.gz'.format(chrom))
		# select_sampled_position_with_MSA_data_one_chrom(args.msa_folder, args.msa_file_suffix, consHMM_df, args.window_around_selected_bp, chrom, sampled_msa_save_fn)
		# print('Done sampling in chromsome {}'.format(chrom))
	print('Done!')