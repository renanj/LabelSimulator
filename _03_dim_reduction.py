 import pandas as pd 
import numpy as np 
from cuml.manifold import TSNE
import multiprocessing
import os

from aux_functions import f_time_now, f_saved_strings, f_log, f_get_files_to_delete, f_delete_files, f_get_subfolders
import config as config
config = config.config

#Inputs:
_script_name = os.path.basename(__file__)
_GPU_flag = config._GPU_Flag_dict[_script_name]

_list_data_sets_path = config._list_data_sets_path
_list_train_val = config._list_train_val



dim_reduction_list = ['t-SNE']


def f_dim_reduction(df, dim_r, n_dimensions=2):
	if dim_r == 't-SNE':
		#colunas X....
		_temp_X_columns = list(df.loc[:,df.columns.str.startswith("X")].columns)
		tsne = TSNE(n_components = n_dimensions, perplexity=100, learning_rate=500)
		X_2dimensions = tsne.fit_transform(df.loc[:,_temp_X_columns])		
		X_2dimensions = X_2dimensions.rename(columns={0: 'X1', 1: 'X2'})
		# X_2dimensions[:,0], X_2dimensions[:,1]		
		df = pd.concat([df[['sample_id',	'name',	'labels',	'manual_label']], X_2dimensions], axis=1)	
		return df

	else:
		print ("We don't have a dim_reduction algo with this name")
  

with open('logs/' + f_time_now(_type='datetime_') + "_03_dim_reduction_py_" + ".txt", "a") as _f:

	_string_log_input = [0, '[INFO] Starting Dimension Reduction']	
	f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)

	_string_log_input = [0, '[INFO] num_cores = ' + str(multiprocessing.cpu_count())]	
	f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)		


	for db_paths in _list_data_sets_path:

		_string_log_input = [1, '[IMAGE DATABASE] = ' + db_paths[0]]	
		f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)


		# _string_log_input = [1, '[INFO] Deleting All Files...']
		# f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)		
		# _sub_folders_to_check = f_get_subfolders(db_paths[0])
		# for _sub_folder in _sub_folders_to_check:	
		#	 f_delete_files(f_get_files_to_delete(_script_name), _sub_folder)		
					
		#folders with DL arch (e.g: VGG_16, VGG_19, etc...)
		_deep_learning_arq_sub_folders =  [db_paths for db_paths in os.listdir(db_paths[4]) if not db_paths.startswith('.')]	

		#remove DL Arc folders with dim_reduction names (e.g.: VGG_16__t-SNE will be removed)
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
						
						#Here Starts the Dim Reduction for dimension list (t-SNE, etc...)
						#Currently we are using only t-SNE
						for dim_r in dim_reduction_list: 

							_string_log_input = [6, 'Dimension = ' + dim_r]	
							f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)
							_string_log_input = [7, 'Exporting .pkl related to = ' + dim_r]	
							f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)

							df_dim = f_dim_reduction(df, dim_r)
							df_dim.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_2D' + _list_train_val[i_train_val] + '.pkl')

							# if '2D' in _file_name:
							# 	df_dim.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_2D_' + _list_train_val[i_train_val] + '.pkl')
							# else:
							# 	df_dim.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_2D' + _list_train_val[i_train_val] + '.pkl')
							

							## Create a folder for each dim reduction algorithm
							# if not os.path.exists(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '__' +  dim_r):
							#   os.makedirs(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '__' +  dim_r)
							# df_dim.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '__' +  dim_r + '/' + 'df_' + _list_train_val[i_train_val] + '.pkl')