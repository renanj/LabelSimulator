import itertools
import multiprocessing
from collections import OrderedDict

import os
import math
import time
import random
import pickle
import warnings
from imutils import paths

from tqdm import tqdm
import concurrent.futures
import multiprocessing
from joblib import Parallel, delayed


import cudf
import cupy as cp
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from baal.active.heuristics import Random, Certainty, Margin, Entropy, Variance, BALD, BatchBALD
from baal.active.heuristics.stochastics import PowerSampling


from config import config
import _05_01_building_blocks as bblocks
from aux_functions import f_time_now, f_saved_strings, f_get_files_to_delete, f_delete_files, f_get_subfolders, try_read_pickle_dataframe, try_pickle_load, try_to_pickle, get_more_recent_file, copy_directory, delete_unlisted_files

warnings.filterwarnings('ignore')


#Inputs:
_script_name = os.path.basename(__file__)
_GPU_flag = config._GPU_Flag_dict[_script_name]

_list_data_sets_path = config._list_data_sets_path
_list_train_val = config._list_train_val

_batch_size_options_config = config._batch_size_options
_batch_size_experiment = config._batch_size_experiment





# Google Drive & Colab sync: 
if config._run_colab_backup_path == True:
    for db_paths in _list_data_sets_path:
        path_colab = config._colab_backup_path
        path_drive = db_paths[0]
        copy_directory(source_path=path_drive, destination_path=path_colab, keep_structure=True)
        print("Saved in Google Colab all Folders and SubFolders from GoogleDrive")  
        # Instructions:
            # 1) First will copy to Colab the Google Drive Folder Structure of tests
            # 2a) When READ Pickle: try it google drive OR colab
            # 2b) When TO Pickle: try it google drive OR colab  




for db_paths in _list_data_sets_path:



    _sub_folders_to_check = f_get_subfolders(db_paths[0])
    for _sub_folder in _sub_folders_to_check:   
        f_delete_files(f_get_files_to_delete(_script_name), _sub_folder)        


    _sub_folders_to_check = f_get_subfolders(config._colab_backup_path + db_paths[0])
    for _sub_folder in _sub_folders_to_check:   
        f_delete_files(f_get_files_to_delete(_script_name), _sub_folder)    


    _deep_learning_arq_sub_folders =  [db_paths for db_paths in os.listdir(db_paths[4]) if not db_paths.startswith('.')]
    for _deep_learning_arq_sub_folder_name in _deep_learning_arq_sub_folders: #vgg_16, #vgg_19,...                                  



        #Open Files!
        try:
            #abrir do google drive
            _df_train = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_train.pkl')
            _df_faiss_indices = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_faiss_indices_train.pkl')
            _df_faiss_distances = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_faiss_distances_train.pkl')
            _df_validation = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_validation.pkl')
            
            _df_2D_train = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_2D_train.pkl')
            _df_2D_faiss_indices = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_2D_faiss_indices_train.pkl')
            _df_2D_faiss_distances = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_2D_faiss_distances_train.pkl')
            _df_2D_validation = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_2D_validation.pkl')
        
        except:
            #abrir do colab
            _df_train = pd.read_pickle(config._colab_backup_path + db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_train.pkl')
            _df_faiss_indices = pd.read_pickle(config._colab_backup_path + db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_faiss_indices_train.pkl')
            _df_faiss_distances = pd.read_pickle(config._colab_backup_path + db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_faiss_distances_train.pkl')
            _df_validation = pd.read_pickle(config._colab_backup_path + db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_validation.pkl')
            
            _df_2D_train = pd.read_pickle(config._colab_backup_path + db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_2D_train.pkl')
            _df_2D_faiss_indices = pd.read_pickle(config._colab_backup_path + db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_2D_faiss_indices_train.pkl')
            _df_2D_faiss_distances = pd.read_pickle(config._colab_backup_path + db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_2D_faiss_distances_train.pkl')
            _df_2D_validation = pd.read_pickle(config._colab_backup_path + db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_2D_validation.pkl')


        try:           
            # _label_encoder = pickle.load(open(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'label_encoder.pkl', 'rb'))           
            _label_encoder = try_pickle_load(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'label_encoder.pkl')             
        except:
            #reading from Colab:
            _label_encoder = try_pickle_load(config._colab_backup_path + db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'label_encoder.pkl')             
            


        _df_train['labels'] = _label_encoder.transform(_df_train['labels'])
        # _df_validation['labels'] = _label_encoder.transform(_df_validation['labels'])         

        print("classes = ", _label_encoder.classes_)
        print("\n\n\n\n\n\n") 


        if config.human_simulations == True:        

            print("Original Space")
            _list_strategy_name, _list_strategy_ordered_samples_id = bblocks.f_run_human_simulations(df_embbedings = _df_train, 
                                                    df_faiss_indices=_df_faiss_indices, 
                                                    df_faiss_distances=_df_faiss_distances)
            _simulation_order_df = pd.DataFrame(_list_strategy_ordered_samples_id).T
            _simulation_order_df.columns = _list_strategy_name  

            print("------------------------------------------")
            print("2D Space")
            _list_strategy_name_2D, _list_strategy_ordered_samples_id_2D = bblocks.f_run_human_simulations(df_embbedings = _df_2D_train, 
                                                    df_faiss_indices=_df_2D_faiss_indices, 
                                                    df_faiss_distances=_df_2D_faiss_distances)
            _simulation_order_df_2D = pd.DataFrame(_list_strategy_ordered_samples_id_2D).T
            _simulation_order_df_2D.columns = _list_strategy_name_2D     



            # _simulation_order_df.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_simulation_order_df.pkl')
            # _simulation_order_df_2D.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_simulation_order_df_2D.pkl')

            try_to_pickle(pickle_file=_simulation_order_df, path=db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_simulation_order_df.pkl')            
            try_to_pickle(pickle_file=_simulation_order_df_2D, path=db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_simulation_order_df.pkl')         

            try_to_pickle(pickle_file=_simulation_order_df, path=config._colab_backup_path +  db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_simulation_order_df.pkl')           
            try_to_pickle(pickle_file=_simulation_order_df_2D, path=config._colab_backup_path +  db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_simulation_order_df.pkl')                        


            
