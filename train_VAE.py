import os, sys
import argparse
import pandas as pd
import numpy as np
import json
import helper
from EVE import VAE_model
from utils import data_utils
import torch
import time
import glob

# msa_data_fn: one file per protein
# MSA_list: a file with columsn protein_name, msa_location, theta (corresponding to the theta_reweighting param, default 0.2)
# MSA_weights_location: empty, potentially an output location
# VAE_checkpoint_folder: empty, potentially an output location
# model_name_suffix
# model_parameters_location: encoder, decoder, training parameters
# training_logs_location: empty, an output location where we will store logs of training

def extract_pretrained_data_from_VAE_checkpoint_folder(VAE_checkpoint_folder, model_name_suffix):
    checkpoint_fn_list = glob.glob(VAE_checkpoint_folder + '/{}_step_*'.format(model_name_suffix))
    epoch_list = list(map(lambda x: int(x.split('_step_')[-1]), checkpoint_fn_list))
    max_epoch = np.max(epoch_list)
    pretrained_model_fn = os.path.join(VAE_checkpoint_folder, '{m}_step_{e}'.format(m = model_name_suffix, e = max_epoch))
    return pretrained_model_fn, max_epoch
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--msa_data_fn', type=str, help='Path to the sampled training data. This has been done by the code create_ideal_training_data.py as a separate step')
    parser.add_argument('--VAE_checkpoint_folder', type=str, help='Location where VAE model checkpoints will be stored')
    parser.add_argument('--model_name_suffix', default='Jan1', type=str, help='model checkpoint name will be the protein name followed by this suffix')
    parser.add_argument('--model_parameters_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--training_logs_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--seed', type=int, default=731995, help='Random seed')
    parser.add_argument('--continue_train_where_left', default = False, action = 'store_true', help='If this flag is set to true, we will find the latest model parameters file in VAE_checkpoint_folder, load the model, and continue training the model where it left off')
    args = parser.parse_args()
    helper.check_file_exist(args.msa_data_fn)
    helper.make_dir(args.VAE_checkpoint_folder)
    helper.make_dir(args.training_logs_location)
    helper.check_file_exist(args.model_parameters_location)
    start_time = time.time()
    torch.manual_seed(args.seed)
    data = data_utils.MSA_processing(
            msa_data_fn=args.msa_data_fn,
            seed=args.seed,
            create_msa_for_evol_indices_calculation = False
    ) # all the attributes of data: data.__dict__.keys()
    print('Done getting input data after: {}'.format(time.time()-start_time))
    model_name = args.model_name_suffix
    print("Model name: "+str(model_name))

    model_params = json.load(open(args.model_parameters_location))
    model_params["training_parameters"]['training_logs_location'] = args.training_logs_location
    model_params["training_parameters"]['model_checkpoint_folder'] = args.VAE_checkpoint_folder

    model = VAE_model.VAE_model(
                    model_name=model_name,
                    data=data,
                    encoder_parameters=model_params["encoder_parameters"],
                    decoder_parameters=model_params["decoder_parameters"],
                    training_parameters=model_params['training_parameters'],
                    random_seed=args.seed
    )
    model = model.to(model.device)


    print("Starting to train model: " + model_name)
    if args.continue_train_where_left:
        pretrained_model_fn, start_epoch = extract_pretrained_data_from_VAE_checkpoint_folder(args.VAE_checkpoint_folder, args.model_name_suffix)
        print('We will restart training model from pretrained weights from {fn}, with epoch {e}'.format(fn = pretrained_model_fn, e = start_epoch))
        model.continue_train_model(data, pretrained_model_fn, start_epoch)
    else:
        model.train_model(data=data)

    print("Saving model: " + model_name)
    model.save(model_checkpoint=model_params["training_parameters"]['model_checkpoint_folder']+os.sep+model_name+"_final", 
                encoder_parameters=model_params["encoder_parameters"], 
                decoder_parameters=model_params["decoder_parameters"], 
                training_parameters=model_params["training_parameters"]
    )