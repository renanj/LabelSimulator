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

from aux_functions import f_time_now, f_saved_strings, f_log, f_create_chart, f_model_accuracy
import building_blocks as bblocks
import config as config
config = config.config




######## WARNING ########

# "index" and "sample_id" are completing different thngs!

########################


#Inputs:
_GPU_flag = config._GPU_Flag_dict['05_simulations.py']

_list_data_sets_path = config._list_data_sets_path
_list_train_val = config._list_train_val


def join_lists_equal_distance(_list_a, _list_b):

    if len(_list_a) <= len(_list_b):
        list_1 = _list_a
        list_2 = _list_b
    else:
        list_1 = _list_b
        list_2 = _list_a        


    #To dynamically insert elements from list_1 into list_2 with equal distance between each insert
    list_1 = cp.array(list_1)
    list_2 = cp.array(list_2)
    # Calculate the spacing between each insert
    spacing = len(list_2) // len(list_1) + 1
    # Generate a range of indices to insert the elements from list_1
    indices = cp.arange(spacing, len(list_2), spacing)
    indices_tuple = tuple(map(int, indices))
    # Slice list_2 into segments between the insertion indices
    segments = [cp.array(list_2[i:j]) for i,j in zip([0]+indices.tolist(), indices.tolist()+[None])]
    # Concatenate the segments with list_1
    result = cp.concatenate([cp.concatenate([s, cp.array([l])]) for s,l in zip(segments, list_1)])
    result = result.tolist()
    return result

    


def f_run_simulations(df_embbedings, df_faiss_indices, df_faiss_distances, simulation_list = None):


    if simulation_list is None:
        # simulation_list = ['Random', 'Equal_Spread', 'Dense_Areas_First', 'Centroids_First', 'Cluster_Boarder_First',  'Outliers_First']
        simulation_list = ['Random', 'Equal_Spread', 'Dense_Areas_First', 'Centroids_First',  'Outliers_First']
        # simulation_list = ['Random', 'Equal_Spread']
    else:
        if 'Random' not in simulation_list:
            simulation_list.append('Random')
        else: 
            None


    #FAISS INDICES & DISTANCES DATAFRAME    
    # df_faiss_indices, df_faiss_distances = bblocks.f_faiss(df_embbedings)

    #BUILDING BLOCKS:
    print("[INFO] -- Creating Building Blocks...")
    print("Random:")
    _samples_id_list_random, _samples_id_list_random_cold_start = bblocks.f_cold_start(df_embbedings)    
    print("SPB:")
    _samples_id_list_ordered_SPB = bblocks.f_SPB(df_embbedings, df_faiss_distances, df_faiss_indices, _cold_start_samples_id=_samples_id_list_random_cold_start)
    print("DEN:")
    _samples_id_list_ordered_DEN = bblocks.f_den(df_embbedings, df_faiss_distances, df_faiss_indices, _cold_start_samples_id=_samples_id_list_random_cold_start, k=5)
    print("OUT:")
    _samples_id_list_ordered_OUT = bblocks.f_out(_samples_id_list_ordered_DEN)
    print("CLU:")
    _centroids_samples_id_list_ordered_CLU, _clusters_samples_id_list_of_lists_ordered_CLU = bblocks.f_clu(df_embbedings, num_clusters=None, num_iterations=25, gpu_index=True)
    print("------------------------------------------------\n\n")


    
    _list_simulations_sample_id = []
    _list_simulations_proceeded = []
    print("[INFO] -- Starting Simulation...")

    #SIMULATION RUN based on "simulation_list":
    for _sim in simulation_list:

        if _sim == 'Random':
            print("Starting Random...")            
            _list_simulations_sample_id.append(_samples_id_list_random)
            _list_simulations_proceeded.append(_sim)
            print("Qtd Samples = ", len(_samples_id_list_random))
            print("End Random!")
            print("--------------------\n")


        elif _sim == 'Equal_Spread':
            print("Starting Equal Spread...")                        
            _samples_id_ordered = _samples_id_list_ordered_SPB
            _list_simulations_sample_id.append(_samples_id_ordered)
            _list_simulations_proceeded.append(_sim)
            print("Qtd Samples = ", len(_samples_id_ordered))
            print("End Equal Spread!")
            print("--------------------\n")


        elif _sim == 'Dense_Areas_First':
            print("Starting Dense_Areas_First...")    
            print("_samples_id_list_ordered_SPB type =", type(_samples_id_list_ordered_SPB))        
            print("_samples_id_list_ordered_DEN type =", type(_samples_id_list_ordered_DEN))                    
            _samples_id_ordered = list(itertools.chain.from_iterable(zip(_samples_id_list_ordered_SPB, _samples_id_list_ordered_DEN)))
            _samples_id_ordered = list(OrderedDict.fromkeys(_samples_id_ordered))
            _list_simulations_sample_id.append(_samples_id_ordered)
            _list_simulations_proceeded.append(_sim)


            _samples_id_ordered = list(itertools.chain.from_iterable(zip(_samples_id_list_ordered_SPB, _samples_id_list_ordered_DEN)))
            _samples_id_ordered = list(OrderedDict.fromkeys(_samples_id_ordered))

            print("Qtd Samples = ", len(_samples_id_ordered))
            print("End Dense_Areas_First!")
            print("--------------------\n")


        elif _sim == 'Centroids_First':
            print("Starting Centroids_First...")                        
            _samples_id_ordered = join_lists_equal_distance(_samples_id_list_ordered_SPB, _centroids_samples_id_list_ordered_CLU)
            _samples_id_ordered = list(OrderedDict.fromkeys(_samples_id_ordered))        
            _list_simulations_sample_id.append(_samples_id_ordered)
            _list_simulations_proceeded.append(_sim)
            print("Qtd Samples = ", len(_samples_id_ordered))
            print("End Centroids_First!")
            print("--------------------\n")    


        # elif _sim == 'Cluster_Borders_First':
        #     print("Starting Cluster_Borders_First'...")            
            
        #     _samples_id_ordered_by_cluster = []
        #     for i in range(len(_clusters_samples_id_list_of_lists_ordered_CLU)):
        #         samples_id_in_cluster = _clusters_samples_id_list_of_lists_ordered_CLU[i]

        #         #SPB:
        #         df_temp = df_embbedings[df_embbedings['sample_id'].isin(samples_id_in_cluster)].copy()
        #         df_temp = df_temp.reset_index(drop=True)

        #         df_faiss_distances_temp = df_faiss_distances[df_faiss_distances.index.isin(samples_id_in_cluster)].copy()
        #         df_faiss_distances_temp = df_faiss_distances_temp.reset_index(drop=True)

        #         df_faiss_indices_temp = df_faiss_distances[df_faiss_distances.index.isin(samples_id_in_cluster)].copy()
        #         df_faiss_indices_temp = df_faiss_indices_temp.reset_index(drop=True)

        #         _temp_samples_id_list_random, _temp_samples_id_list_random_cold_start = bblocks.f_cold_start(df_temp)    
        #         _temp_samples_id_list_ordered_SPB = bblocks.f_SPB(df_temp, df_faiss_distances_temp, df_faiss_indices_temp, _cold_start_samples_id=_temp_samples_id_list_random_cold_start)

        #         #OUT:
        #         _temp_samples_id_list_ordered_DEN = bblocks.f_den(df_temp, df_faiss_distances_temp, df_faiss_indices_temp, _cold_start_samples_id=_temp_samples_id_list_random_cold_start, k=5)
        #         _temp_samples_id_list_ordered_OUT = bblocks.f_out(_temp_samples_id_list_ordered_DEN)                


        #         #SPB(50%) + OUT(50%) within Cluster:
        #         _temp_list = [val for pair in zip(_temp_samples_id_list_ordered_SPB, _temp_samples_id_list_ordered_OUT) for val in pair]
        #         _temp_list = list(set(_temp_list))                
        #         _samples_id_ordered_by_cluster.append(_temp_list)

            
            # _samples_id_ordered = [val for pair in zip(*_samples_id_ordered_by_cluster) for val in pair]
            # print("Len List = ", len(_samples_id_ordered))
            # _samples_id_ordered = list(set(_samples_id_ordered))                
            # print("Len List = ", len(_samples_id_ordered))

            # _list_simulations_sample_id.append(_samples_id_ordered)
            # _list_simulations_proceeded.append(_sim)
            # print("Qtd Samples = ", len(_samples_id_ordered))
            # print("End Cluster_Borders_First'!")
            # print("--------------------")    


        elif _sim == 'Outliers_First':
            print("Starting Outliers_First...")                        
            _samples_id_ordered = list(itertools.chain.from_iterable(zip(_samples_id_list_ordered_SPB, _samples_id_list_ordered_OUT)))
            _samples_id_ordered = list(OrderedDict.fromkeys(_samples_id_ordered))
            _list_simulations_sample_id.append(_samples_id_ordered)
            _list_simulations_proceeded.append(_sim)
            print("Qtd Samples = ", len(_samples_id_ordered))
            print("End Outliers_First!")
            print("--------------------\n")

        
        else:
            print("We don't have a function ready for {} simulation!", _sim)
    return _list_simulations_proceeded, _list_simulations_sample_id            




with open(f_time_now(_type='datetime_') + "logs/05_simulations_py_" + ".txt", "a") as _file:


    _string_log_input = [0, '[INFO] Starting Simulations']    
    f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)


    _string_log_input = [0, '[INFO] num_cores = ' + multiprocessing.cpu_count()]    
    f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)



    for db_paths in _list_data_sets_path:

        _string_log_input = [1, '[IMAGE DATABASE] = ' + db_paths[0]]    
        f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)

        _deep_learning_arq_sub_folders =  [db_paths for db_paths in os.listdir(db_paths[4]) if not db_paths.startswith('.')]
        for _deep_learning_arq_sub_folder_name in _deep_learning_arq_sub_folders:            

            _string_log_input = [2, 'Architecture ' + _deep_learning_arq_sub_folder_name]    
            f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)

            _list_files = [_file_name for _files in os.listdir(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name) if not _file_name.startswith('.')]        

            _string_log_input = [3, 'List of Files = ' + _list_files]    
            f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)


            
            _list_files_temp = []
            for _file_name in _list_files:
                if _file_name !='df_'+ _list_train_val[i_train_val] + '.pkl':                    
                else:                                        
                    _list_files_temp.append(_file_name)
            _list_files = None
            _list_files = _list_files_temp.copy()


            _string_log_input = [3, 'line_split_01']    
            f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)

            
            for i_train_val in range(len(_list_train_val)):                            

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

                        df = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder + '/' + _files)
                        df_faiss_indices = pd.read_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder + '/' + 'df_faiss_indices_' + _list_train_val[i_train_val]  + '.pkl')
                        df_faiss_distances = pd.read_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder + '/' + 'df_faiss_distances_' + _list_train_val[i_train_val]  + '.pkl')

                        _string_log_input = [5, '[INFO] Starting Simulations']    
                        f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)                        

                        
                        _list_simulation_sample_name, _list_simulation_ordered_samples_id = f_run_simulations(df_embbedings = df, df_faiss_indices=df_faiss_indices, df_faiss_distances=df_faiss_distances, simulation_list = None)                        

                        _string_log_input = [6, 'Exporting .pkl related to = ' + dim_r]    
                        f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)                        



                        _simulation_order_df = pd.DataFrame(_list_simulation_ordered_samples_id).T
                        _simulation_order_df.columns = _list_simulation_sample_name                
                        _simulation_order_df.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder + '/' + 'df_simulation_ordered_' + _list_train_val[i_train_val]  + '.pkl')