import os
import sys
import random
import time
import datetime
import math

import pandas as pd
import numpy as np
import seaborn as sns
from imutils import paths

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from sklearn.exceptions import DataConversionWarning, ConvergenceWarning

from tqdm import tqdm
import concurrent.futures
import multiprocessing
from joblib import Parallel, delayed

# import cudf
# import cuml

from aux_functions import f_time_now, f_saved_strings, f_log, f_create_accuracy_chart, f_create_visualization_chart_animation, f_get_files_to_delete, f_delete_files, f_get_subfolders
# f_generate_gif_chart_scatterplots
import config as config
config = config.config




#Inputs:
_script_name = os.path.basename(__file__)
_GPU_flag = config._GPU_Flag_dict[_script_name]
_list_data_sets_path = config._list_data_sets_path
_list_train_val = config._list_train_val





def f_model_accuracy(_args):

	_df, _model, _ordered_samples_id, _qtd_samples_to_train, _GPU_flag = _args
	
	_ordered_samples_id_temp = _ordered_samples_id[0:_qtd_samples_to_train+1]
	# print("LEN == ", len(_ordered_samples_id_temp))
	
	if _GPU_flag is True:
		_temp_X_columns = [x for x, mask in zip(_df.columns.values, _df.columns.str.startswith("X")) if mask]
		X_train = _df[_df['sample_id'].isin(_ordered_samples_id_temp)].loc[:,_temp_X_columns].astype('float32')
		y_train = _df[_df['sample_id'].isin(_ordered_samples_id_temp)].loc[:,'labels'].astype('float32')	   
		X_test = _df.loc[:,_temp_X_columns].astype('float32')
		y_test = _df.loc[:,'labels'].astype('float32')

	else:		
		# print("TPU")															
		_temp_X_columns = list(_df.loc[:,_df.columns.str.startswith("X")].columns)																
		X_train = _df[_df['sample_id'].isin(_ordered_samples_id_temp)].loc[:,_temp_X_columns]
		y_train = _df[_df['sample_id'].isin(_ordered_samples_id_temp)].loc[:,'labels']					  
		X_test = _df.loc[:,_temp_X_columns]
		y_test = _df.loc[:,'labels']
		# print("X_train .shape = ", X_train.shape)
		# print("X_test .shape = ", X_test.shape)


	try:					
		_model.fit(X_train, y_train)									
		_score = _model.score(X_test, y_test)
		# print("worked for..", _qtd_samples_to_train)
		# print("_score = ", _score)
		# print("\n\n")
		return _score
	
	except:											
		_score = -1
		print("entered i expection...", i)
		print("\n\n")
		return _score


_list_paths = [
	'/content/drive/MyDrive/Mestrado/Git/LabelSimulator/data/plancton/db_feature_extractor/vgg_16__t-SNE',
	'/content/drive/MyDrive/Mestrado/Git/LabelSimulator/data/plancton/db_feature_extractor/vgg_19__t-SNE'
]

_list_df_simulations = [
	'/content/drive/MyDrive/Mestrado/Git/LabelSimulator/data/plancton/db_feature_extractor/vgg_16__t-SNE/df_framework_train.pkl',
	'/content/drive/MyDrive/Mestrado/Git/LabelSimulator/data/plancton/db_feature_extractor/vgg_19__t-SNE/df_framework_train.pkl'
] 


_list_df_simulations_ordered = [
	'/content/drive/MyDrive/Mestrado/Git/LabelSimulator/data/plancton/db_feature_extractor/vgg_16__t-SNE/df_simulation_samples_ordered_train.pkl',
	'/content/drive/MyDrive/Mestrado/Git/LabelSimulator/data/plancton/db_feature_extractor/vgg_19__t-SNE/df_simulation_samples_ordered_train.pkl'	
] 

_list_dfs = [
	'/content/drive/MyDrive/Mestrado/Git/LabelSimulator/data/plancton/db_feature_extractor/vgg_16__t-SNE/df_train.pkl',
	'/content/drive/MyDrive/Mestrado/Git/LabelSimulator/data/plancton/db_feature_extractor/vgg_19__t-SNE/df_train.pkl'
]



						


for i in range(len(_list_dfs)):

	df_simulation = pd.read_pickle(_list_df_simulations[i])
	df = pd.read_pickle(_list_dfs[i])
	_df_simulation_ordered = pd.read_pickle(_list_df_simulations_ordered[i])

	_list_simulation_sample_name = list(_df_simulation_ordered.columns) 
	_list_simulation_ordered_samples_id = _df_simulation_ordered.T.values.tolist()										 

	f_create_accuracy_chart(df_simulation, 
					_path=_list_paths[i] + '/' + 'vis_accuracy_chart_' + 'train' + '.png')


	f_create_visualization_chart_animation(
		_df_2D = df, 
		_path=_list_paths[i], 
		_file_name = 'vis_2D_selection_' + 'train',
		_list_simulation_names=_list_simulation_sample_name,
		_list_selected_samples= _list_simulation_ordered_samples_id,
		_n_fractions=5, _fps=3)					 




[TO-DO]

1. Deletar pastas de DL Arc (vgg_16, vgg_19) quando rodar o delete_files