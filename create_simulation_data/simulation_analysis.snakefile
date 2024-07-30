import pandas as pd
import os
configfile: './config/config.yaml'
raw_emission_fn = config['raw_emission_fn']
consHMM_state_reorder_fn = config['consHMM_state_reorder_fn']
correct_emission_fn = config['correct_emission_fn']
simulation_folder = config['simulation_folder']
train_seed = config['train_seed']
test_seed = config['test_seed']
consHMM_state_to_simulate = config['consHMM_state_to_simulate']
num_bp_per_state_to_simulate = config['num_bp_per_state_to_simulate']
refSpec_name = config['refSpec_name']
sample_simulate_msa_fn = config['sample_simulate_msa_fn']
pretrain_model_fn = config['pretrain_model_fn']
model_parameters_location = config['model_parameters_location']

rule all:
	input:
		os.path.join(simulation_folder, 'train_data', 'train_data.txt.gz'),
		os.path.join(simulation_folder, 'train_data', 'test_data.txt.gz'),
		os.path.join(simulation_folder, 'evol_indices', 'test_data.txt.gz')

rule process_raw_consHMM_emission:
#. first. we need to create the correct consHMM emission matrix so that we can sample simulation data based on ConsHMM parameters
	input:
		raw_emission_fn,
		consHMM_state_reorder_fn
	output:
		correct_emission_fn
	shell:
		"""
		python postProcess_consHMM_emission.py --raw_emission_fn {raw_emission_fn} --state_reorder_fn {consHMM_state_reorder_fn} --output_fn {correct_emission_fn}
		"""

rule create_simulation_data:
	input:
		correct_emission_fn
	output:
		os.path.join(simulation_folder, 'train_data', 'train_data.txt.gz'),
		os.path.join(simulation_folder, 'train_data', 'test_data.txt.gz')
	params:
		train_fn_no_gz = os.path.join(simulation_folder, 'train_data', 'train_data.txt'),
		test_fn_no_gz = os.path.join(simulation_folder, 'train_data', 'test_data.txt'),
		consHMM_state_to_simulate = ' '.join(list(map(lambda x: str(x), consHMM_state_to_simulate))),
	shell:
		'''
		python $PWD/create_simulation_data_from_consHMM.py  --emission_fn {correct_emission_fn} --output_fn {params.train_fn_no_gz} --state_list {params.consHMM_state_to_simulate} --num_bp_per_state {num_bp_per_state_to_simulate} --refSpec_name {refSpec_name} --sample_output_fn {sample_simulate_msa_fn} --seed {train_seed}
		python $PWD/create_simulation_data_from_consHMM.py --emission_fn {correct_emission_fn} --output_fn {params.test_fn_no_gz} --state_list {params.consHMM_state_to_simulate} --num_bp_per_state {num_bp_per_state_to_simulate} --refSpec_name {refSpec_name} --sample_output_fn {sample_simulate_msa_fn} --seed {test_seed}
		'''

rule calculate_evol_indices:
	input:
		os.path.join(simulation_folder, 'train_data', 'test_data.txt.gz'),
		pretrain_model_fn
	output:
		os.path.join(simulation_folder, 'evol_indices', 'test_data.txt.gz'),
	params:
		batch_size = 4096, # batch_size for calculating the evol_indices
		num_samples_compute_evol_indices = 10, # # sample z that we will sample to compute the evol_indices for each genetic variants
	shell:
		'''
		python /u/home/h/havu73/project-ernst/source_eve_rep/ncEVE/compute_evol_indices.py \
    	--msa_data_fn {input[0]} \
    	--VAE_pretrained_fn {pretrain_model_fn} \
    	--model_parameters_location {model_parameters_location} \
    	--output_evol_indices_fn {output} \
    	--num_samples_compute_evol_indices {params.num_samples_compute_evol_indices} \
    	--batch_size {params.batch_size}
		'''

# rule create_json_model_params:
# 	input:
# 	output:
	