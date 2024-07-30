import numpy as np
import pandas as pd
from collections import defaultdict
import os
import torch
import tqdm

def compute_weight(seq):
    number_non_empty_positions = np.dot(seq,seq)
    if number_non_empty_positions>0:
        denom = np.dot(list_seq,seq) / np.dot(seq,seq) 
        denom = np.sum(denom > 1 - self.theta) 
        return 1/denom
    else:
        return 0.0 #return 0 weight if sequence is fully empty

class MSA_processing:
    def __init__(self,
        msa_data_fn="",
        seed = 731995,
        create_msa_for_evol_indices_calculation = False,
        refSpec_colname = 'hg19'
        ):
        
        """
        Parameters:
        - msa_data_fn: (path) Location of the MSA data, sampled using consHMM data, outputted by create_ideal_training_data.py
        - seed: random seed, for reproducibility 
        - msa_file_suffix: quite redudant right now but let's keep it at that
        - create_msa_for_evol_indices_calculation: if set to True, it will declare an attribute genPos_df that records the genome position in msa_data_fn. This will be useful when we record the evol_score for genetic variants
        """
        self.seed = seed
        torch.manual_seed(self.seed)
        self.msa_data_fn = msa_data_fn # this file should be the results of the code create_ideal_training_data.py
        self.create_msa_for_evol_indices_calculation = create_msa_for_evol_indices_calculation
        self.refSpec_colname = refSpec_colname
        self.alphabet = "ACTGXN"
        self.alphabet_size = len(self.alphabet)
        self.refGen_alphabet = self.alphabet[:4]
        self.get_nu_dict()
        self.chrom_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 'X', 'Y'] 
        if self.create_msa_for_evol_indices_calculation:
            self.create_single_mutation_msa_data() # declare self.msa_df, self.genPos_df, self.num_genomic_position, self.num_species, self.reference_species_index
        else:
            self.read_in_msa_for_training() # read in data of multi-species sequence alignment
            # declare self.msa_df, self.genPos_df, self.num_genomic_position, self.num_species, self.reference_species_index
        print('Done reading in input tensor. Done declaring data')


    def get_nu_dict(self):
        self.nu_dict = {}
        for i,nu in enumerate(self.alphabet):
            self.nu_dict[nu] = int(i)
        return 


    def read_raw_msa(self):
        self.msa_df = pd.read_csv(self.msa_data_fn, header = 0, index_col = None, sep = '\t', comment = '#') 
        self.msa_df = self.msa_df[self.msa_df[self.refSpec_colname] != 'N']
        try: # this case happens if the data is produced for trainings
            self.msa_df.rename({'chosen_bp': 'bp'}, axis = 1, inplace = True)
            # self.msa_df.drop(columns = ['state'], axis = 'columns', inplace = True)
        except: 
            pass
        try: # this case happens if the data is produced for getting evol_indices
            # if self.create_msa_for_evol_indices_calculation == True:
            self.genPos_df = self.msa_df[['chrom', 'bp', 'state', self.refSpec_colname]] # refSpec_colname is likly hg19 for our project 
            self.msa_df.drop(columns = ['chrom', 'bp', 'state'], axis = 'columns', inplace = True) # columns are simply names of different species #HAHAHAHA: edit this chrom',
        except:
            pass
        self.msa_df = self.msa_df.applymap(lambda x: self.nu_dict[x]) # a dataframe of just integers, converting the msa data into numbers just like [genPos, species]
        self.num_genomic_position = self.msa_df.shape[0]
        self.num_species = self.msa_df.shape[1] # number of columns: number of species
        self.reference_species_index = self.msa_df.columns.get_loc(self.refSpec_colname) # column index of the reference species (most likely hg19). This parameter is helpful when we try to create data of all types of SNPs' multi-species alignment, and hes to be declared here before we change the data into pytorch tensor and the colnames will disappear
        return 

    def read_in_msa_for_training(self):
        self.read_raw_msa() # declare self.msa_df, self.genPos_df, self.num_genomic_position, self.num_species, self.reference_species_index
        self.msa_df = torch.tensor(self.msa_df.values) # convert  the dataframe to a pytorch tensor, [position][species]
        return 

    def create_single_mutation_msa_data(self):
        self.read_raw_msa() # declare self.msa_df, self.genPos_df, self.num_genomic_position, self.num_species, self.reference_species_index
        num_repeat_per_nu = len(self.refGen_alphabet) # 4
        self.msa_df = self.msa_df.loc[self.msa_df.index.repeat(num_repeat_per_nu)].reset_index(drop = True) # for each row in the original msa_df, repeat num_repeat_per_nu times such that every num_repeat_per_nu consecutive rows are exactly similar. This would blow up the size of msa_df num_repeat_per_nu times. Reference: https://stackoverflow.com/questions/49074021/repeat-rows-in-data-frame-n-times
        refSpec_variants = list(range(num_repeat_per_nu)) * self.num_genomic_position # [0,1,2,3, 0,1,2,3, 0,1,2,3, etc.] --> record all forms of genetic variants for each genomic position
        self.msa_df.loc[:, self.refSpec_colname] = refSpec_variants
        self.msa_df = torch.tensor(self.msa_df.values) # convert  the dataframe to a pytorch tensor, [position][species]
        return 

    def one_hot_encoding_seq(self):
        '''
        Given the size of the data, I am not sure if it is a good idea to call on this function. Keep it here for documentation purposes
        '''
        self.msa_df = torch.nn.functional.one_hot(self.msa_df, num_classes = self.alphabet_size) # [position][species][nucleotide]

    def __getitem__(self, index):
        return torch.nn.functional.one_hot(self.msa_df[index], num_classes = self.alphabet_size) # here index refers to the genomic positoin, so the output should be of size #_animal, #_alphabet

    def __len__(self):
        return self.msa_df.shape[0] # return number of bp in the dataset