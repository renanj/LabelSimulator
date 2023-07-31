import pandas as pd 
import numpy as np 
from cuml.manifold import TSNE
import multiprocessing
import os
import optuna

from aux_functions import f_time_now, f_saved_strings, f_log, f_get_files_to_delete, f_delete_files, f_get_subfolders

# import config as config
# config = config.config

import config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('test_number')
args = parser.parse_args()
config = config.Config(args.test_number)


#Inputs:
_script_name = os.path.basename(__file__)
_GPU_flag = config._GPU_Flag_dict[_script_name]

_list_data_sets_path = config._list_data_sets_path
_list_train_val = config._list_train_val


dim_reduction_list = ['t-SNE']



def scoring_function(x):    
    return np.random.rand()

def objective(trial, df):  
    n_dimensions = 2
    perplexity = trial.suggest_int('perplexity', 5, 50)
    learning_rate = trial.suggest_loguniform('learning_rate', 10, 1000)
    n_iter = trial.suggest_int('n_iter', 1000, 5000)
    _temp_X_columns = list(df.loc[:,df.columns.str.startswith("X")].columns)
    tsne = TSNE(n_components=n_dimensions, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
    X_2dimensions = tsne.fit_transform(df.loc[:, _temp_X_columns])
    score = scoring_function(X_2dimensions)
    return score

def f_dim_reduction(df, n_trials=50):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, df), n_trials=n_trials)
    best_params = study.best_params
    _temp_X_columns = list(df.loc[:,df.columns.str.startswith("X")].columns)

    tsne = TSNE(n_components=2, perplexity=best_params['perplexity'], learning_rate=best_params['learning_rate'], n_iter=best_params['n_iter'])  # n_components is fixed to 2

    X_2dimensions = tsne.fit_transform(df.loc[:, _temp_X_columns])
    X_2dimensions = X_2dimensions.rename(columns={0: 'X1', 1: 'X2'})                    
    #X_2dimensions = pd.DataFrame(X_2dimensions, columns=['X1', 'X2'])
    df = pd.concat([df[['sample_id', 'name', 'labels', 'manual_label']], X_2dimensions], axis=1)
    return df





with open('logs/' + f_time_now(_type='datetime_') + "_03_dim_reduction_py_" + ".txt", "a") as _f:

	_string_log_input = [0, '[INFO] Starting Dimension Reduction']	
	f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)

	_string_log_input = [0, '[INFO] num_cores = ' + str(multiprocessing.cpu_count())]	
	f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)		


	i_dim_per = 0
	for db_paths in _list_data_sets_path:

		_string_log_input = [1, '[IMAGE DATABASE] = ' + db_paths[0]]	
		f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)

		_deep_learning_arq_sub_folders =  [db_paths for db_paths in os.listdir(db_paths[4]) if not db_paths.startswith('.')]	
		
		for item_sub_folder in _deep_learning_arq_sub_folders:
			for item_dim_r in dim_reduction_list:
				if item_dim_r in item_sub_folder:					
					_deep_learning_arq_sub_folders.remove(item_sub_folder)


		for _deep_learning_arq_sub_folder_name in _deep_learning_arq_sub_folders:			

			_list_files = [_files for _files in os.listdir(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name) if not _files.startswith('.')]

			_string_log_input = [2, 'Architecture ' + _deep_learning_arq_sub_folder_name]	
			f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)					

			_string_log_input = [3, 'List of Files = ' + str(_list_files)]	
			f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)
			_string_log_input = [3, 'line_split_01']	
			f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)			


			for i_train_val in range(len(_list_train_val)):							

				_string_log_input = [4, '[RUN] ' + _list_train_val[i_train_val]]	
				f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)				


				for _file_name in _list_files:			 
					if _file_name not in ['df_'+ _list_train_val[i_train_val] + '.pkl']: 										
						None
					else:
						_string_log_input = [4, 'Running File = ' + _file_name]	
						f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)
						_string_log_input = [5, '[INFO] Starting Dim Reduction']	
						f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)

						df = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + _file_name) 
						


						# _string_log_input = [6, 'Dimension = ' + dim_r]	
						# f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)
						# _string_log_input = [6, 'Exporting .pkl related to = ' + dim_r]	
						# f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)

						df_dim = f_dim_reduction(df)
						df_dim.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_2D_' + _list_train_val[i_train_val] + '.pkl')


		i_dim_per = i_dim_per + 1							