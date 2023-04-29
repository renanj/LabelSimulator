import pandas as pd 
import numpy as np 
from cuml.manifold import TSNE
import os

import config as config
config = config.config

dim_reduction_list = ['t-SNE']


def f_dim_reduction(df, dim_r, n_dimensions=2):
  if dim_r == 't-SNE':
    #colunas X....
    _temp_X_columns = list(df.loc[:,df.columns.str.startswith("X")].columns)
    tsne = TSNE(n_components = n_dimensions)
    X_2dimensions = tsne.fit_transform(df.loc[:,_temp_X_columns])        
    X_2dimensions = X_2dimensions.rename(columns={0: 'X1', 1: 'X2'})
    # X_2dimensions[:,0], X_2dimensions[:,1]        
    df = pd.concat([df[['sample_id',	'name',	'labels',	'manual_label']], X_2dimensions], axis=1)
    	
    return df        
    
  else:
    print ("We don't have a dim_reduction algo with this name")
    
  

with open(f_time_now(_type='datetime_') + "logs/03_dim_reduction_py_" + ".txt", "a") as f:

    _string_log_input = ['[INFO] Starting Dimension Reduction', 0]    
    f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
    f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))            

    _string_log_input = ['[INFO] num_cores = ' + multiprocessing.cpu_count(), 0]    
    f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
    f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))            


    for db_paths in _list_data_sets_path:

        _string_log_input = ['[IMAGE DATABASE] = ' + db_paths[0], 1]
        f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
        f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))        

        _deep_learning_arq_sub_folders =  [db_paths for db_paths in os.listdir(db_paths[4]) if not db_paths.startswith('.')]
        for _deep_learning_arq_sub_folder_name in _deep_learning_arq_sub_folders:            

            _string_log_input = ['Architecture ' + _deep_learning_arq_sub_folder_name, 2]    
            f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
            f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))                              

            _list_files = [_file_name for _files in os.listdir(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name) if not _file_name.startswith('.')]        


            _string_log_input = ['List of Files = ' + _list_files, 3]    
            f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
            f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))                              

            
            _list_files_temp = []
            for _file_name in _list_files:
                if _file_name !='df_'+ _list_train_val[i_train_val] + '.pkl':                    
                else:                                        
                    _list_files_temp.append(_file_name)
            _list_files = None
            _list_files = _list_files_temp.copy()

            _string_log_input = ['line_split_01', 3]    
            f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
            f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))                              
            
            
            for i_train_val in range(len(_list_train_val)):                            

                _string_log_input = ['[RUN] ' + _list_train_val[i_train_val], 4]                    
                f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
                f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))

                for _file_name in _list_files:             
                    if _file_name !='df_'+ _list_train_val[i_train_val] + '.pkl':
                        #f_print(' ' * 6 + 'Aborting... File not valid for this run!' + '\n\n', _level=4)
                        None
                    else:
                        _string_log_input = ['Running File = ' + _file_name, 4]    
                        f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
                        f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))


                        #Here Starts the Dim Reduction for Each DB                    
                        df = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder + '/' + _file_name) 


                        _string_log_input = ['[INFO] Starting Dim Reduction', 5]    
                        f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
                        f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))                        
                        
                        for dim_r in dim_reduction_list:  

                            _string_log_input = ['Dimension = ' + dim_r, 6]    
                            f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
                            f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))                                                    

                            df = f_dim_reduction(df, dim_r)  

                            _string_log_input = ['Exporting .pkl related to = ' + dim_r, 7]    
                            f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
                            f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))                                                    

                            if not os.path.exists(db_paths[4] +'/' + _deep_learning_arq_sub_folder + '__' +  dim_r):
                              os.makedirs(db_paths[4] +'/' + _deep_learning_arq_sub_folder + '__' +  dim_r)
                            df.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder + '__' +  dim_r + '/' + 'df_' + _list_train_val[i_train_val] + '.pkl')
                        