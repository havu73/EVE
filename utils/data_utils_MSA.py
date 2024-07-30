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
        MSA_location="",
        theta=0.2,
        use_weights=True,
        weights_location="./data/weights",
        preprocess_MSA=True,
        threshold_sequence_frac_gaps=0.5,
        threshold_focus_cols_frac_gaps=0.3,
        remove_sequences_with_indeterminate_AA_in_focus_cols=True
        ):
        
        """
        Parameters:
        - msa_location: (path) Location of the MSA data. Constraints on input MSA format: 
            - focus_sequence is the first one in the MSA data
            - first line is structured as follows: ">focus_seq_name/start_pos-end_pos" (e.g., >SPIKE_SARS2/310-550)
            - corespondding sequence data located on following line(s)
            - then all other sequences follow with ">name" on first line, corresponding data on subsequent lines
        - theta: (float) Sequence weighting hyperparameter. Generally: Prokaryotic and eukaryotic families =  0.2; Viruses = 0.01
        - use_weights: (bool) If False, sets all sequence weights to 1. If True, checks weights_location -- if non empty uses that; 
            otherwise compute weights from scratch and store them at weights_location
        - weights_location: (path) Location to load from/save to the sequence weights
        - preprocess_MSA: (bool) performs pre-processing of MSA to remove short fragments and positions that are not well covered.
        - threshold_sequence_frac_gaps: (float, between 0 and 1) Threshold value to define fragments
            - sequences with a fraction of gap characters above threshold_sequence_frac_gaps are removed
            - default is set to 0.5 (i.e., fragments with 50% or more gaps are removed)
        - threshold_focus_cols_frac_gaps: (float, between 0 and 1) Threshold value to define focus columns
            - positions with a fraction of gap characters above threshold_focus_cols_pct_gaps will be set to lower case (and not included in the focus_cols)
            - default is set to 0.3 (i.e., focus positions are the ones with 30% of gaps or less, i.e., 70% or more residue occupancy)
        - remove_sequences_with_indeterminate_AA_in_focus_cols: (bool) Remove all sequences that have indeterminate AA (e.g., B, J, X, Z) at focus positions of the wild type
        """
        np.random.seed(2022)
        self.MSA_location = MSA_location
        self.weights_location = weights_location
        self.theta = theta
        self.alphabet = "ACTGX" # note X means no alignment
        self.use_weights = use_weights
        self.preprocess_MSA = preprocess_MSA
        self.threshold_sequence_frac_gaps = threshold_sequence_frac_gaps
        self.threshold_focus_cols_frac_gaps = threshold_focus_cols_frac_gaps
        self.remove_sequences_with_indeterminate_AA_in_focus_cols = remove_sequences_with_indeterminate_AA_in_focus_cols

        self.gen_alignment() # first will do some preprocessing of the sequences, filter out some sequences or columns that do not reach quality control (too much gaps), and then do one-hot encoding of sequences, and then calculate weight of each sequence
        self.create_all_singles() # create a list of all the possible mutation to the focus sequence of protein. Declare self.mutant_to_letter_pos_idx_focus_list and self.all_single_mutations

    def get_bp_dict(self):
        self.bp_dict = {}
        for i,aa in enumerate(self.alphabet):
            self.bp_dict[aa] = i
        return 

    def first_read_MSA_location(self):
        self.seq_name_to_sequence = defaultdict(str)# keys: sequence names (in lines starting with >), values: sequence
        name = ""
        with open(self.MSA_location, "r") as msa_data:
            for i, line in enumerate(msa_data):
                line = line.rstrip()
                if line.startswith(">"):
                    name = line
                    if i==0:
                        self.focus_seq_name = name
                else:
                    self.seq_name_to_sequence[name] += line
        return 

    def preprocess_MSA_func(self):
        msa_df = pd.DataFrame.from_dict(self.seq_name_to_sequence, orient='index', columns=['sequence']) # df with index: protein sequence name, 1 column which is the sequence
        # Data clean up
        msa_df.sequence = msa_df.sequence.apply(lambda x: x.replace(".","-")).apply(lambda x: ''.join([aa.upper() for aa in x])) # convert all sequences to upper case , -- stands foralignment gap
        # Remove columns that would be gaps in the wild type
        non_gap_wt_cols = [aa!='-' for aa in msa_df.sequence[self.focus_seq_name]] # list of boolean with length == len(focus sequence), each corresponding to a aa, True if it is NOT -
        msa_df['sequence'] = msa_df['sequence'].apply(lambda x: ''.join([aa for aa,non_gap_ind in zip(x, non_gap_wt_cols) if non_gap_ind])) # for each of the other sequences, we filter out aa that align with the gap (-) in the focus sequence
        assert 0.0 <= self.threshold_sequence_frac_gaps <= 1.0,"Invalid fragment filtering parameter"
        assert 0.0 <= self.threshold_focus_cols_frac_gaps <= 1.0,"Invalid focus position filtering parameter"
        msa_array = np.array([list(seq) for seq in msa_df.sequence]) # array of list, each list is a list of aa in a sequence
        gaps_array = np.array(list(map(lambda seq: [aa=='-' for aa in seq], msa_array))) # array of list of boolean values. Each list shows T/F values for each aa in each seq, indicating whether that position is a gap in the sequence
        # Identify fragments with too many gaps --> shape [#seq, #aa_per_seq]
        seq_gaps_frac = gaps_array.mean(axis=1)
        seq_below_threshold = seq_gaps_frac <= self.threshold_sequence_frac_gaps # array of T/F, length = #seq --> whether that sequence has acceptable gap frequency
        print("Proportion of sequences dropped due to fraction of gaps: "+str(round(float(1 - seq_below_threshold.sum()/seq_below_threshold.shape)*100,2))+"%")
        # Identify focus columns (first only look at sequences that satify the tolerable gap threshold, then within those sequences, lok at each column (each aa) and see if that position satisfy that it has tolerable gaps))
        columns_gaps_frac = gaps_array[seq_below_threshold].mean(axis=0)
        index_cols_below_threshold = columns_gaps_frac <= self.threshold_focus_cols_frac_gaps
        print("Proportion of non-focus columns removed: "+str(round(float(1 - index_cols_below_threshold.sum()/index_cols_below_threshold.shape)*100,2))+"%")
        # Lower case non focus cols (cols with higher than tolerable gap threshold, and keep uppercase ow). Then filter out sequences with too much gaps
        msa_df['sequence'] = msa_df['sequence'].apply(lambda x: ''.join([aa.upper() if upper_case_ind else aa.lower() for aa, upper_case_ind in zip(x, index_cols_below_threshold)]))
        msa_df = msa_df[seq_below_threshold]
        # Overwrite seq_name_to_sequence with clean version
        self.seq_name_to_sequence = defaultdict(str)
        for seq_idx in range(len(msa_df['sequence'])):
            self.seq_name_to_sequence[msa_df.index[seq_idx]] = msa_df.sequence[seq_idx]
        return 

    def get_seq_name_to_sequence(self):
        self.first_read_MSA_location() # declare seq_name_to_sequence: dictionary keys: sequence name (species), values: sequences
        ## MSA pre-processing to remove inadequate columns and sequences
        if self.preprocess_MSA:
            self.preprocess_MSA_func()

        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
        self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper() and s!='-'] # list of indices of bases in the focus sequence that are one of those bases with a lot of gaps across the different species (sequences)
        self.focus_seq_trimmed = [self.focus_seq[ix] for ix in self.focus_cols] # focus seq with only well-aligned bases
        self.seq_len = len(self.focus_cols) # only include bases with tolerable gaps
        self.alphabet_size = len(self.alphabet)

        # Connect local sequence index with uniprot index (index shift inferred from 1st row of MSA)
        focus_loc = self.focus_seq_name.split("/")[-1]
        start,stop = focus_loc.split("-")
        self.focus_start_loc = int(start)
        self.focus_stop_loc = int(stop)
        self.uniprot_focus_col_to_wt_aa_dict \
            = {idx_col+int(start):self.focus_seq[idx_col] for idx_col in self.focus_cols} 
        self.uniprot_focus_col_to_focus_idx \
            = {idx_col+int(start):idx_col for idx_col in self.focus_cols} 

        # Move all letters to CAPS; filter out all columns of sequences that are not among the focus columns
        for seq_name,sequence in self.seq_name_to_sequence.items():
            sequence = sequence.replace(".","-") # a bit redundant, but actually it's because this code has to work for whether we had preprocess_MSA_func or not
            self.seq_name_to_sequence[seq_name] = [sequence[ix].upper() for ix in self.focus_cols]
        return 

    def remove_sequences_with_indeterminate_AA_in_focus_cols_func(self):
        # Remove sequences that have indeterminate AA (e.g., B, J, X, Z) in the focus columns
        if self.remove_sequences_with_indeterminate_AA_in_focus_cols:
            alphabet_set = set(list(self.alphabet))
            seq_names_to_remove = []
            for seq_name,sequence in self.seq_name_to_sequence.items():
                for letter in sequence:
                    if letter not in alphabet_set and letter != "-":
                        seq_names_to_remove.append(seq_name)
                        continue
            seq_names_to_remove = list(set(seq_names_to_remove))
            for seq_name in seq_names_to_remove:
                del self.seq_name_to_sequence[seq_name]
        return 

    def one_hot_encoding_seq(self):
        '''
        after having self.seq_name_to_sequence all fixed and cleaned, we will do one-hot-encoding of the sequence into self.one_hot_encoding. Note to Ha: this can actually be done using numpy built-in one-hot-encoding function
        '''
        print ("Encoding sequences")
        self.one_hot_encoding = np.zeros((len(self.seq_name_to_sequence.keys()),len(self.focus_cols),len(self.alphabet))) # 3D array of zeros
        for i,seq_name in enumerate(self.seq_name_to_sequence.keys()):
            sequence = self.seq_name_to_sequence[seq_name]
            for j,letter in enumerate(sequence):
                if letter in self.bp_dict: # if the flag self.remove_sequences_with_indeterminate_AA_in_focus_cols, then this check is actually redundant, but this is here to check for all cases
                    k = self.bp_dict[letter]
                    self.one_hot_encoding[i,j,k] = 1.0
        return 

    def calculate_seq_weights(self):
        if self.use_weights:
            try:
                self.weights = np.load(file=self.weights_location)
                print("Loaded sequence weights from disk")
            except:
                print ("Computing sequence weights")
                list_seq = self.one_hot_encoding
                list_seq = list_seq.reshape((list_seq.shape[0], list_seq.shape[1] * list_seq.shape[2]))
                self.weights = np.array(list(map(compute_weight,list_seq)))
                np.save(file=self.weights_location, arr=self.weights)
        else:
            # If not using weights, use an isotropic weight matrix
            print("Not weighting sequence data")
            self.weights = np.ones(self.one_hot_encoding.shape[0])
        return 

    def gen_alignment(self):
        """ Read training alignment and store basics in class instance """
        self.get_bp_dict() # declare the self.bp_dict: keys: aa, value: index of the aa in the alphabet
        self.get_seq_name_to_sequence() # declare the self.seq_nam_to_sequence, keys: sequence name (in lines starting with >), values: sequence (potentially with cleaning that filter out parts of the sequences with too many gaps)
        self.remove_sequences_with_indeterminate_AA_in_focus_cols_func() # this function will remove all those sequences in self.seq_name_to_sequence that have indeterminate aa within self.focus_cols

        # Encode the sequences
        self.one_hot_encoding_seq() # will declare self.one_hot_encoding [seq_name/index, column, letter in the aa alphabet]

        # calculate sequence weights, have not looked at how these weigths are calculated yet but not top priority right now
        self.calculate_seq_weights() # declare self.weights: 1D array [sequence]

        self.Neff = np.sum(self.weights)
        self.num_sequences = self.one_hot_encoding.shape[0]

        print ("Neff =",str(self.Neff))
        print ("Data Shape =",self.one_hot_encoding.shape)
    
    def create_all_singles(self):
        start_idx = self.focus_start_loc
        focus_seq_index = 0
        self.mutant_to_letter_pos_idx_focus_list = {}
        list_valid_mutations = []
        # find all possible valid mutations that can be run with this alignment
        alphabet_set = set(list(self.alphabet))
        for i,letter in enumerate(self.focus_seq):
            if letter in alphabet_set and letter != "-":
                for mut in self.alphabet:
                    pos = start_idx+i
                    if mut != letter:
                        mutant = letter+str(pos)+mut
                        self.mutant_to_letter_pos_idx_focus_list[mutant] = [letter, pos, focus_seq_index]
                        list_valid_mutations.append(mutant)
                focus_seq_index += 1   
        self.all_single_mutations = list_valid_mutations # list of all the possible mutations from the focus sequence. Format for each entry: letter(in focus sequence), index of the letter in the focus sequence, mutant letter (any other letter that's not in the focus letter). Note: no space. 

    def save_all_singles(self, output_filename):
        with open(output_filename, "w") as output:
            output.write('mutations')
            for mutation in self.all_single_mutations:
                output.write('\n')
                output.write(mutation)