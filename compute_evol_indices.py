import os,sys
import json
import argparse
import pandas as pd
import torch
import helper
from EVE import VAE_model
from utils import data_utils
def calculate_evol_indices_one_row(row, refSpec_colname, all_bases):
    '''
    refSpec_colname: colname representing the reference species, showing the bp (ACTG) that are present in the refSpec's genome at the corresponding location
    all_bases: ['A', 'C', 'T', 'G']
    '''
    row[all_bases] = row[all_bases] - row[row[refSpec_colname]]
    return row

def extract_evol_indices_from_model_output(msa_data, mean_elbos, output_evol_indices_fn, command_line):
    '''
    mean_elbos: (num_genomic_position, 4 (ACTG))
    msa_data: created with the create_msa_for_evol_indices_calculation = True --> contains the msa_data.genPos_df with columns: chrom, chosen_bp, state, hg19
    '''
    result_df = msa_data.genPos_df.copy() # chrom, bp, hg19 (hg19 shows the bp at the corresponding position)
    all_bases = list(msa_data.refGen_alphabet)
    mean_elbos = pd.DataFrame(mean_elbos) # (num_genomic_position, 4 (ACTG)). Have to tranform to df because the following line would crash otherwise
    result_df[all_bases] = mean_elbos
    # result_df = result_df.apply(lambda row: calculate_evol_indices_one_row(row, msa_data.refSpec_colname, all_bases), axis = 1)
    helper.save_gzip_dataframe_with_comment(result_df, command_line, output_evol_indices_fn)
    return 

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Evol indices')
    parser.add_argument('--msa_data_fn', type=str, help='Path to the sampled training data. This has been done by the code create_ideal_training_data.py as a separate step')
    parser.add_argument('--VAE_pretrained_fn', type=str, help='Location where pre-trained VAE model checkpoints will be stored')
    parser.add_argument('--model_parameters_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--seed', type=int, default=731995, help='Random seed')
    parser.add_argument('--output_evol_indices_fn', type=str, help='Output location of computed evol indices')
    parser.add_argument('--num_samples_compute_evol_indices', type=int, help='Num of samples to approximate delta elbo when computing evol indices')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size when computing evol indices')
    parser.add_argument('--compute_recon_prob', action = 'store_true', default = False, help = 'If this flag is set, we will compute the reconstruction probabilities for each letter (ACTGXN) at each base')
    args = parser.parse_args()
    helper.check_file_exist(args.msa_data_fn)
    helper.check_file_exist(args.VAE_pretrained_fn)
    helper.check_file_exist(args.model_parameters_location)
    helper.create_folder_for_file(args.output_evol_indices_fn)
    command_line = helper.print_command(sys.argv[0], args)

    torch.manual_seed(args.seed)
    data = data_utils.MSA_processing(
            msa_data_fn=args.msa_data_fn,
            seed=args.seed,
            create_msa_for_evol_indices_calculation = (not args.compute_recon_prob), 
    ) # all the attributes of data: data.__dict__.keys()
    print ("Done loading all input data")

    model_params = json.load(open(args.model_parameters_location))
    model_name= 'trial'
    model = VAE_model.VAE_model(
                    model_name=model_name,
                    data=data,
                    encoder_parameters=model_params["encoder_parameters"],
                    decoder_parameters=model_params["decoder_parameters"],
                    training_parameters=model_params['training_parameters'],
                    random_seed=args.seed,
                    pretrained_model = True)
    model = model.to(model.device)

    model.load_pretrained_model(args.VAE_pretrained_fn)
    print ("Done loading pre-trained VAE model")

    mean_elbos = model.compute_reconstruction_prob_for_variants_data(msa_data=data,
                            num_samples=args.num_samples_compute_evol_indices,
                            batch_size=args.batch_size) # (num_genomic_position, 4 (ACTG))
    print('Done calculating the mean_elbos for all variants')

    # now we will report the results to a file
    extract_evol_indices_from_model_output(data, mean_elbos, args.output_evol_indices_fn, command_line)
    print('Done reporting the results into file {}'.format(args.output_evol_indices_fn))