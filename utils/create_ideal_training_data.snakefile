import os
configfile: 'config.yaml'
msa_folder = config['msa_folder']
msa_file_suffix = config['msa_file_suffix'] 
consHMM_folder = config['consHMM_folder']
sampledReg_folder = config['sampledReg_folder']
subsample_folder = config['subsample_folder']
window_around_selected_bp = config['window_around_selected_bp']
seed = config['seed']
num_bp_per_state = config['num_bp_per_state']
num_consHMM_state = config['num_consHMM_state']
chrom_length_fn = config['chrom_length_fn']
CHROMOSOME_LIST = config['CHROMOSOME_LIST']
num_cores = config['num_cores']

rule all:
	input:
		os.path.join(subsample_folder, 'sampled_msa_gw.txt.gz'),
		
rule sample_region:
	input:
	output:
		expand(os.path.join(sampledReg_folder, 'chr{chrom}_sampled_bp.txt.gz'), chrom = CHROMOSOME_LIST)
	shell:
		"""
		python create_ideal_training_data.py --msa_folder {msa_folder} --msa_file_suffix {msa_file_suffix} --consHMM_folder {consHMM_folder} --output_folder {sampledReg_folder} --window_around_selected_bp {window_around_selected_bp} --seed {seed} --num_bp_per_state {num_bp_per_state} --num_consHMM_state {num_consHMM_state} --chrom_length_fn {chrom_length_fn}
		"""

rule join_msa_data_and_sampled_regions:
	input:
		os.path.join(sampledReg_folder, 'chr{chrom}_sampled_bp.txt.gz'),
		os.path.join(msa_folder, 'chr{chrom}_'+msa_file_suffix), # msa_fn
	output:
		expand(os.path.join(sampledReg_folder, 'chr{{chrom}}_{core}_sampled_msa.txt.gz'), core = range(num_cores)),
	params:
		output_folder = sampledReg_folder
	shell:
		'''
		msa_fn_length=$(zcat {input[1]} | wc -l)
		python join_msa_data_and_sampled_regions.py  --msa_fn {input[1]}  --sampledReg_folder {sampledReg_folder} --output_folder {params.output_folder} --chrom {wildcards.chrom} --msa_fn_length ${{msa_fn_length}} --num_cores {num_cores}
		gzip -f {params.output_folder}/chr{wildcards.chrom}_*_sampled_msa.txt
 		'''

rule combine_all_msa_sampled_region_data:
	input:
		expand(os.path.join(sampledReg_folder, 'chr{chrom}_{core}_sampled_msa.txt.gz'), chrom = CHROMOSOME_LIST, core = range(num_cores)),
	output:
		os.path.join(sampledReg_folder, 'sampled_msa_gw.txt.gz'),
	params:
		output_no_gz = os.path.join(sampledReg_folder, 'sampled_msa_gw.txt')
	shell:
		"""
		for f in {sampledReg_folder}/chr*_*_sampled_msa.txt.gz
		do

			if [ ! -f {params.output_no_gz} ]
			then
				zcat $f >> {params.output_no_gz}
			else
				zcat $f | tail -n +2 >> {params.output_no_gz} # skip the first line because it is header line. Reference: https://stackoverflow.com/questions/604864/print-a-file-skipping-the-first-x-lines-in-bash
			fi
		done
		gzip {params.output_no_gz}
		"""

rule downsample_msa_data:
	input:
		expand(os.path.join(sampledReg_folder, 'chr{chrom}_{core}_sampled_msa.txt.gz'), chrom = CHROMOSOME_LIST, core = range(num_cores)),
	output:
		os.path.join(subsample_folder, 'sampled_msa_gw.txt.gz')
	params:
		output_no_gz = os.path.join(subsample_folder, 'sampled_msa_gw.txt')
	shell:
		'''
		for f in {sampledReg_folder}/chr*_*_sampled_msa.txt.gz
		do
			if [ ! -f {params.output_no_gz} ]
			then
				zcat $f | sed -n 1p >> {params.output_no_gz} # write the header line if the output file has not been created
			fi
			num_line=$(zcat $f | tail -n +2 | wc -l)
			num_sample=$(( ($num_line+9)/10 )) # sample 10% of the positions recorded in this file
			zcat $f | tail -n +2 | shuf -n ${{num_sample}} >> {params.output_no_gz} 
			# Note: 
			# tail -n +2 : skip the first line 
			# round up division in bash: https://stackoverflow.com/questions/2395284/round-a-divided-number-in-bash
			# subsample N lines from file: https://stackoverflow.com/questions/9245638/select-random-lines-from-a-file
		done
		gzip {params.output_no_gz}
		'''