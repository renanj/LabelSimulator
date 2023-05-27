import pandas as pd
import numpy as np
import os
import random
import math
import itertools
import multiprocessing
from collections import OrderedDict
from re import S

from scipy.spatial import distance_matrix
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

import sys
import cudf
import cupy as cp

from aux_functions import f_time_now, f_saved_strings, f_log, f_get_files_to_delete, f_delete_files, f_get_subfolders
import _05_01_building_blocks as bblocks
import _05_02_active_learning as active_learning_query
import config as config
config = config.config



######## WARNING ########

# "index" and "sample_id" are completing different thngs!

########################


#Inputs:
_script_name = os.path.basename(__file__)
_GPU_flag = config._GPU_Flag_dict[_script_name]

_list_data_sets_path = config._list_data_sets_path
_list_train_val = config._list_train_val


with open('logs/' + f_time_now(_type='datetime_') + "_05_simulations_py_" + ".txt", "a") as _f:

	_string_log_input = [0, '[INFO] Starting Simulations']	
	f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)


	_string_log_input = [0, '[INFO] num_cores = ' + str(multiprocessing.cpu_count())]	
	f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)



	for db_paths in _list_data_sets_path:

		_string_log_input = [1, '[IMAGE DATABASE] = ' + db_paths[0]]	
		f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)


		_string_log_input = [1, '[INFO] Deleting All Files...']
		f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)		
		_sub_folders_to_check = f_get_subfolders(db_paths[0])
		for _sub_folder in _sub_folders_to_check:	
			f_delete_files(f_get_files_to_delete(_script_name), _sub_folder)		
					

		_deep_learning_arq_sub_folders =  [db_paths for db_paths in os.listdir(db_paths[4]) if not db_paths.startswith('.')]
		for _deep_learning_arq_sub_folder_name in _deep_learning_arq_sub_folders:			

			_list_files = [_files for _files in os.listdir(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name) if not _files.startswith('.')]
			

			_string_log_input = [2, 'Architecture ' + _deep_learning_arq_sub_folder_name]	
			f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)
			_string_log_input = [3, 'List of Files = ' + str(_list_files)]	
			f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)
			_string_log_input = [3, 'line_split_01']	
			f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)

			
			for i_train_val in range(len(_list_train_val)):				
				if config._list_train_val[i_train_val] == 'validation':
					continue										

				_string_log_input = [4, '[RUN] ' + _list_train_val[i_train_val]]	
				f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)

				for _file_name in _list_files:			 
					if _file_name !='df_'+ _list_train_val[i_train_val] + '.pkl':
						#f_print(' ' * 6 + 'Aborting... File not valid for this run!' + '\n\n', _level=4)
						None
					else:
						_string_log_input = [4, 'Running File = ' + _file_name]	
						f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)						

						###Start Simulations:

						df = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + _file_name)
						df_faiss_indices = pd.read_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_faiss_indices_' + _list_train_val[i_train_val]  + '.pkl')
						df_faiss_distances = pd.read_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_faiss_distances_' + _list_train_val[i_train_val]  + '.pkl')
						df_validation = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_validation.pkl')



						_string_log_input = [5, '[INFO] Starting Simulations']	
						f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)						

						
						_list_simulation_sample_name, _list_simulation_ordered_samples_id = bblocks.f_run_human_simulations(df_embbedings = df, df_faiss_indices=df_faiss_indices, df_faiss_distances=df_faiss_distances, _human_simulation_list = None)
						_list_active_learning_query_name, _list_active_learning_query_ordered_samples_id = active_learning_query.f_run_active_learning(df_embbedings = df,  _df_validation= df_validation)


						_list_strategy_name = _list_simulation_sample_name.copy()
						_list_strategy_name.extend(_list_active_learning_query_name)

						_list_strategy_ordered_samples_id = _list_simulation_ordered_samples_id.copy()
						_list_strategy_ordered_samples_id.extend(_list_active_learning_query_ordered_samples_id)						


						_string_log_input = [6, 'Exporting .pkl']	
						f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)						

						_simulation_order_df = pd.DataFrame(_list_strategy_ordered_samples_id).T
						_simulation_order_df.columns = _list_strategy_name				
						_simulation_order_df.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_simulation_samples_ordered_' + _list_train_val[i_train_val]  + '.pkl')