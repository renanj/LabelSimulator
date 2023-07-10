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

from aux_functions import f_time_now, f_saved_strings, f_log, f_create_consolidate_accuracy_chart, f_create_visualization_chart_animation, f_get_files_to_delete, f_delete_files, f_get_subfolders, f_create_random_vs_query_accuracy_chart
# f_generate_gif_chart_scatterplots
import config as config
config = config.config


#Inputs:
_script_name = os.path.basename(__file__)
_GPU_flag = config._GPU_Flag_dict[_script_name]
_list_data_sets_path = config._list_data_sets_path
_list_train_val = config._list_train_val


def generate_output_list(df):

    order_dict = {}


    for index, row in df.iterrows():
        query_strategy = row['Query_Strategy']
        if query_strategy not in order_dict:
            order_dict[query_strategy] = len(order_dict)


    sorted_query_strategy = sorted(order_dict, key=order_dict.get)


    output_df = df.groupby('Query_Strategy')['Samples IDs'].agg(sum).reset_index()


    output_list_of_list = [
        sorted_query_strategy,
        [output_df.loc[output_df['Query_Strategy'] == query_strategy, 'Samples IDs'].tolist()[0] for query_strategy in sorted_query_strategy]
    ]

    return output_list_of_list


with open('logs/' + f_time_now(_type='datetime_') + "_06_generate_visualization_py_" + ".txt", "a") as _f:

    _string_log_input = [0, '[INFO] Starting Framework']	
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


        _string_log_input = [1, '[INFO] Chart Creation']	
        f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)        
        
        _deep_learning_arq_sub_folders =  [db_paths for db_paths in os.listdir(db_paths[4]) if not db_paths.startswith('.')]
        for _deep_learning_arq_sub_folder_name in _deep_learning_arq_sub_folders:									

        
            _df_framework = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_framework.pkl')
            _df_2D_train = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_2D_train.pkl')
            _df_2D_validation = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_2D_validation.pkl')



            #Chart 1 -- Consolidate Accuracy Chart
            f_create_consolidate_accuracy_chart(_df_framework, 
                            _path=db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'vis_01_consolidate_accuracy_chart_' + 'train' + '.png',
                            _file_name = 'vis_01_consolidate_accuracy_chart_' + 'train'
                            _col_x = 'Percetage Samples Evaluated',
                            _col_y = 'Samples Accuracy Train',
                            _hue='Query_Strategy')

            f_create_consolidate_accuracy_chart(_df_framework, 
                            _path=db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'vis_01_consolidate_accuracy_chart_' + 'validation' + '.png',
                            _file_name = 'vis_01_consolidate_accuracy_chart_' + 'validation',
                            _col_x = 'Percetage Samples Evaluated',
                            _col_y = 'Samples Accuracy Validation',
                            _hue='Query_Strategy')
            


            #Chart 2 -- Accuracy Chart Strategy vs. Random
            f_create_random_vs_query_accuracy_chart(_df_framework, 
                            _path=db_paths[4] +'/' + _deep_learning_arq_sub_folder_name,
                            _file_name = 'vis_02_accuracy_vs_random_chart_' + 'train',
                            _col_x = 'Percetage Samples Evaluated',
                            _col_y = 'Samples Accuracy Train',
                            _hue='Query_Strategy')
            
            f_create_random_vs_query_accuracy_chart(_df_framework, 
                            _path=db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'vis_02_accuracy_vs_random_chart_' + 'validation' + '.png',
                            _file_name = 'vis_02_accuracy_vs_random_chart_' + 'validation',
                            _col_x = 'Percetage Samples Evaluated',
                            _col_y = 'Samples Accuracy Validation',
                            _hue='Query_Strategy')            


            #Chart 3 -- Accuracy Delta vs. Random

            #Chart 4 -- 2D Selection Evolution Chart & 2D & Accuracy Evolution Gif

            
            unique_query_strategies = _df_framework['Query_Strategy'].unique().tolist()                        
            query_strategies_samples_id = [_df_framework[_df_framework['Query_Strategy'] == query_strategy].groupby('Query_Strategy')['Samples IDs'].sum().tolist() for query_strategy in unique_query_strategies]


            f_create_visualization_chart_animation(
                _df_2D = _df_2D_train, 
                _path=db_paths[4] +'/' + _deep_learning_arq_sub_folder_name, 
                _file_name = 'vis_04_selection_2D' + _list_train_val[i_train_val],
                _list_simulation_names= unique_query_strategies,
                _list_selected_samples= query_strategies_samples_id
                _n_fractions=5, _fps=3)            
            