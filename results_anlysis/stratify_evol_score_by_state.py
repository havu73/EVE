from pybedtools import BedTool
import pandas as pd 
import numpy as np 
import os 
import sys 
import argparse
import helper
import glob
import seaborn as sns

parser = argparse.ArgumentParser(description='Calculate the average evol_index from ncEVE, stratified by consHMM state. This is an exploratory analysis for the results of the model')
parser.add_argument('--evol_index_fn', type=str, required=True,
                    help='File showing the data of evol_index for select genomic regions')
parser.add_argument('--consHMM_fn', type=str, required=True,
                    help='Bed file of consHMM state, usually one chromosome per file')
parser.add_argument('--output_folder', type = str, required = True,
	help = "output_folder")

def read_evol_index_fn(evol_index_fn, consHMM_fn):
evolI_df = pd.read_csv(evol_index_fn, header = 0, index_col = None, sep = '\t')
evolI_df.rename(columns = {'bp': 'start'}, inplace = True)
evolI_df.dropna(axis = 0, how = 'any', inplace = True)
evolI_df['end'] = evolI_df['start'] + 1  
evolI_df = evolI_df[['chrom', 'start', 'end', 'hg19', 'A', 'C', 'T', 'G']]
evolI_bed = BedTool.from_dataframe(evolI_df)
consHMM_bed = BedTool(consHMM_fn)
evolI_bed = evolI_bed.intersect(consHMM_bed, wa =True, wb= True)
evolI_bed = evolI_bed.to_dataframe()
evolI_bed.drop(['itemRgb', 'blockCount', 'blockSizes'], axis = 1, inplace = True)
evolI_bed.rename(columns = {'name': 'hg19', 'score': 'A', 'strand': 'C', 'thickStart': 'T', 'thickEnd': 'G', 'blockStarts': 'state'}, inplace = True)
evol_score_df = evolI_bed[['A', 'C', 'T', 'G']]
evolI_bed['min_nz_score'] = evol_score_df.apply(lambda row: np.min(row[row!=0]), axis = 1)
evolI_bed['max_nz_score'] = evol_score_df.apply(lambda row: np.max(row[row!=0]), axis = 1)
evolI_bed['mean_nz_score'] = evol_score_df.apply(lambda row: np.mean(row[row!=0]), axis=1)

sum_df = evolI_bed.groupby('state').agg({'min_nz_score' : 'mean', 'max_nz_score': 'mean'})


if __name__ == '__main__':
	args = parser.parse_args()
	print(args)
