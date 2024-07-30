export EVE_folder="/u/home/h/havu73/project-ernst/ncEVE_project"
export msa_data_fn="${EVE_folder}/training_data/simulation/test_sim_data.txt.gz"
export VAE_checkpoint_folder="${EVE_folder}/models/simulation/VAE_parameters"
export VAE_pretrained_fn="${VAE_checkpoint_folder}/simulation_step_210000"
export model_name_suffix="simulation"
export model_parameters_location="/u/home/h/havu73/project-ernst/source_eve_rep/ncEVE/EVE/start_model_params.json"
export output_evol_indices_fn="${EVE_folder}/models/simulation/evol_indices/test_sim_data.txt.gz"
export num_samples_compute_evol_indices=10
export batch_size=4096

python /u/home/h/havu73/project-ernst/source_eve_rep/ncEVE/compute_evol_indices.py \
    --msa_data_fn ${msa_data_fn} \
    --VAE_pretrained_fn ${VAE_pretrained_fn} \
    --model_parameters_location ${model_parameters_location} \
    --output_evol_indices_fn ${output_evol_indices_fn} \
    --num_samples_compute_evol_indices ${num_samples_compute_evol_indices} \
    --batch_size ${batch_size}
