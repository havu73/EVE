import pandas as pd 
import numpy as np 
import os 
import sys 
import argparse
import helper
import glob
import seaborn as sns
from sklearn.decomposition import PCA
parser = argparse.ArgumentParser(description='Calculate the average evol_index from ncEVE, stratified by consHMM state. This is an exploratory analysis for the results of the model')
parser.add_argument('--evol_index_fn', type=str, required=True,
                    help='File showing the data of evol_index for select genomic regions')
parser.add_argument('--output_fn', type = str, required = True,
	help = "output_fn")

def pca_score_by_state(evol_index_fn, output_fn):
	evolI_df = pd.read_csv(evol_index_fn, header = 0, index_col = None, sep = '\t', comment = '#')
	evolI_df.dropna(axis = 0, how = 'any', inplace = True)
	pca = PCA(n_components=2)
	pca_df = pca.fit_transform(evolI_df[['A', 'C', 'T', 'G']])
	evolI_df.loc[:, ['pca_1', 'pca_2']] = pd.DataFrame(pca_df, columns = ['pca_1', 'pca_2'])
	print('Done calculating PCA! :)')
	plot = sns.scatterplot(data = evolI_df, x = 'pca_1', y = 'pca_2', hue = 'state', alpha = 0.2)
	fig = plot.get_figure()
	fig.savefig(output_fn)
	print('Done plotting! :)')
	return

if __name__ == '__main__':
	args = parser.parse_args()
	print(args)
	helper.check_file_exist(args.evol_index_fn)
	helper.create_folder_for_file(args.output_fn)
	pca_score_by_state(args.evol_index_fn, args.output_fn)
