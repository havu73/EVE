import pandas as pd 
import numpy as np 
import os 
import argparse
import helper
import multiprocessing as mp 
parser = argparse.ArgumentParser(description="Find positive pairs of variants")
parser.add_argument("--msa_fn", type=str, required=True, help="Folder where each chrom's MSA data is store")
parser.add_argument("--sampledReg_folder", type = str, required=True, help = 'Where we save the data of regions that we sample for training.')
parser.add_argument('--output_folder', type = str, required = True, help='Where data of joined MSA and sampled regions are saved, each file contains output data for one chromosome.')
parser.add_argument('--chrom', required = True, help = 'chromosome that we will join the MSA and the sampled regions. Format: 1, not chr1')
parser.add_argument('--NUM_ROW_PER_SEGMENT', type = int, required = False, default = 500000, help = 'Number of rows that we read from the msa file for each time we open it. The reason why we need this is because this file is too big to be opened all at once')
parser.add_argument('--msa_fn_length', type = int, required = True, help = 'Number of lines of the msa_fn. This can be queried as part of the snakemake pipeline using zcat \{msa_fn\} \| wc -l. This number is useful because eventually we want to be able to divide the job into subprocesses. Note: this number includes one line of header')
parser.add_argument('--num_cores', type = int, required = False, default = 4, help = 'number of parallel cores that we will divide the process into, to make it run faster')
parser.add_argument('--write_df_freq', type = int, required = False, default = 9, help = 'number of segments of the msa file that we will store at once until we save the output to output_fn. We implemented this function so that at once time we do not hold too much data into memory and we also do keep writing to files all the time, which slow down the program')


def read_msa_df_one_chunk(msa_fn, start_row, NUM_ROW_PER_SEGMENT, msa_colnames):
	try:
		msa_df = pd.read_csv(msa_fn, header = None, index_col = 0, skiprows = start_row, nrows = NUM_ROW_PER_SEGMENT, sep = ',')
		msa_df.columns = msa_colnames
		return msa_df
	except:
		print('The reading of file {f} broke when we started at row {r}'.format(f = msa_fn, r = start_row))
		return None

def write_append_to_output(output_fn, result_df_list):
	if len(result_df_list) > 0:
		result_df = pd.concat(result_df_list, axis = 0, ignore_index = True) # concat all the rows
		result_df.to_csv(output_fn, header = not os.path.isfile(output_fn), index = False, sep = '\t', mode = 'a') # write the header if this file has not been created, and NO headers otherwise
	return 

def one_process_select_sampled_position_with_MSA_data(msa_fn, sampledReg_folder, chrom, output_fn, NUM_ROW_PER_SEGMENT, start_row, num_segment, write_df_freq):
	'''
	chrom is assumed to be of the form '1' and not 'chr1'
	sampled_df only contains data for one chromosome
	'''
	print('Hello!')
	sampled_fn = os.path.join(sampledReg_folder, 'chr{c}_sampled_bp.txt.gz'.format(c = chrom))
	sampled_df = pd.read_csv(sampled_fn, header = 0, index_col = None, sep = '\t') #chrom, chosen_bp, state
	msa_colnames = list(pd.read_csv(msa_fn, header= 0, index_col = 0, nrows = 0, sep = ',').columns)
	result_colnames = ['chrom', 'chosen_bp', 'state'] + msa_colnames
	result_df_list = []
	for segment_index in range(num_segment):
		msa_df = read_msa_df_one_chunk(msa_fn = msa_fn, start_row = start_row + segment_index * NUM_ROW_PER_SEGMENT, NUM_ROW_PER_SEGMENT = NUM_ROW_PER_SEGMENT, msa_colnames = msa_colnames)
		msa_df = sampled_df.merge(msa_df, left_on = 'chosen_bp', right_index = True, how = 'inner') # intersection between two dataframes
		result_df_list.append(msa_df)
		if (segment_index + 1) % write_df_freq == 0:  # write to the output file in chunk, so that we do not hold too much memory at one time that cause the program to crash prematurely, and we also do not keep writing into files too often
			write_append_to_output(output_fn, result_df_list)
			result_df_list = []
			print('Done appending to file {f} after segment {i}'.format(f = output_fn, i = segment_index))
	write_append_to_output(output_fn, result_df_list)
	return 

def find_num_segment_four_cores(num_segment, NUM_ROW_PER_SEGMENT):
	num_segment_per_core_list = list(map(lambda x: np.ceil(num_segment*x).astype(int), [0.4, 0.3, 0.2, 0.1])) # the first job  will get done fastest because it traces through the first part  of the file, much faster to open and trace the beggining compared to the end of a file
	culm_num_segment_per_core = [0] + list(np.cumsum(num_segment_per_core_list))
	num_segment_per_core_list[-1] = num_segment_per_core_list[-1] - (culm_num_segment_per_core[-1] - num_segment)
	start_row_list = list(map(lambda x: int(1+ x*NUM_ROW_PER_SEGMENT), culm_num_segment_per_core[:-1])) # each number in the list shows the start_row of the core's job
	return num_segment_per_core_list, start_row_list

def all_processes_select_sampled_position_with_MSA_data(msa_fn, sampledReg_folder, chrom, output_folder, NUM_ROW_PER_SEGMENT, msa_fn_length, num_cores, write_df_freq):
	num_segment = np.ceil((msa_fn_length - 1) / NUM_ROW_PER_SEGMENT).astype(int)
	num_segment_per_core_list, start_row_list = find_num_segment_four_cores(num_segment, NUM_ROW_PER_SEGMENT)
	output_fn_list = list(map(lambda x: os.path.join(output_folder, 'chr{c}_{p}_sampled_msa.txt'.format(c = chrom, p = x)), range(num_cores)))
	processes = [mp.Process(target = one_process_select_sampled_position_with_MSA_data, args = (msa_fn, sampledReg_folder, chrom, output_fn_list[x], NUM_ROW_PER_SEGMENT, start_row_list[x], num_segment_per_core_list[x], write_df_freq)) for x in range(num_cores)] 
	for p in processes:
		p.start()
	for i, p in enumerate(processes):
		p.join()
		print("Process " + str(i) + " is finished!")
	print ('Done!')

if __name__ == '__main__':
	args = parser.parse_args()
	helper.check_file_exist(args.msa_fn)
	helper.check_dir_exist(args.sampledReg_folder)
	helper.make_dir(args.output_folder)
	output_fn = os.path.join(args.output_folder, 'chr{c}_{p}_sampled_msa.txt'.format(c = args.chrom, p = 0))
	start_row = 1
	num_segment = np.ceil((args.msa_fn_length - 1) / args.NUM_ROW_PER_SEGMENT).astype(int)
	print(args.msa_fn_length)
	print(args.num_cores)
	# one_process_select_sampled_position_with_MSA_data(args.msa_fn, args.sampledReg_folder, args.chrom, output_fn, args.NUM_ROW_PER_SEGMENT, start_row, num_segment, args.write_df_freq)
	all_processes_select_sampled_position_with_MSA_data(args.msa_fn, args.sampledReg_folder, args.chrom, args.output_folder, args.NUM_ROW_PER_SEGMENT, args.msa_fn_length, args.num_cores, args.write_df_freq)
	print("Done joining data for chromosome {}".format(args.chrom))
	print ('Done!')