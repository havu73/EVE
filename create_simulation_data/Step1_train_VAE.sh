export EVE_folder="/u/home/h/havu73/project-ernst/ncEVE_project"
export msa_data_fn="${EVE_folder}/training_data/simulation/train_data.txt.gz"
export VAE_checkpoint_folder="${EVE_folder}/models/simulation/VAE_parameters"
export model_name_suffix="simulation"
export model_parameters_location="/u/home/h/havu73/project-ernst/source_eve_rep/ncEVE/EVE/model_params_Bayse.json"
export training_logs_location="${EVE_folder}/models/simulation/logs"

# msa_data_fn: one file per protein
# MSA_list: a file with columsn protein_namem msa_location, theta (not sure what theta is)
# MSA_weights_location: empty, potentially an output location
# VAE_checkpoint_folder: empty, potentially an output location
# model_name_suffix
# model_parameters_location: encoder, decoder, training parameters
# training_logs_location: empty, an output location where we will store logs of training

mkdir -p ${training_logs_location}

command="python /u/home/h/havu73/project-ernst/source_eve_rep/ncEVE/train_VAE.py \
    --msa_data_fn ${msa_data_fn} \
    --VAE_checkpoint_folder ${VAE_checkpoint_folder} \
    --model_name_suffix ${model_name_suffix} \
    --model_parameters_location ${model_parameters_location} \
    --training_logs_location ${training_logs_location} \
    --continue_train_where_left"

echo $command > ${training_logs_location}/command.logs
eval $command
    
