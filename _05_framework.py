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
from aux_functions import f_time_now, f_saved_strings, f_get_files_to_delete, f_delete_files, f_get_subfolders

warnings.filterwarnings('ignore')


#Inputs:
_script_name = os.path.basename(__file__)
_GPU_flag = config._GPU_Flag_dict[_script_name]

_list_data_sets_path = config._list_data_sets_path
_list_train_val = config._list_train_val

_batch_size_options_config = config._batch_size_options
_batch_size_experiment = config._batch_size_experiment




def f_framework_df(
    _df_train, 
	_df_validation, 
	_cold_start_samples_id, 
    _legend_name,
	_query_strategy_name,
	_query_batch_size,   
    _database_name,
    _dl_architecture_name,	
	_df_faiss_indices=None,
	_df_faiss_distances=None,
	_list_ordered_samples_id=None,
    _input_framework_id=None):

    
    _ensembles_heuristics_list = config._ensembles_heuristics_list

    if _query_strategy_name in  _ensembles_heuristics_list:
        _is_ensemble_model = True
    else:
        _is_ensemble_model = False


    if _is_ensemble_model == True:
        base_estimator = LogisticRegression(multi_class='ovr')
        _model = BaggingClassifier(base_estimator=base_estimator, n_estimators=10, random_state=42)
        _model_name = "LogisticRegression (Ensemble)"
    else:
        _model = LogisticRegression(random_state=0)
        _model_name = "LogisticRegression"


    def f_predict(test, clf):        
        x = np.array(list(map(lambda e: e.predict_proba(test[0]), clf.estimators_)))        
        x = np.rollaxis(x, 0, 3) # Roll axis because Baal expect [n_samples, n_classes, ..., n_estimations]
        return x        

    
    if _query_strategy_name == "Uncertainty":
        _al_function = Certainty()

    elif _query_strategy_name == "Entropy":
        _al_function = Entropy()

    elif _query_strategy_name == "Margin":
        _al_function = Margin()

    elif _query_strategy_name == "Bald":
        _al_function = BALD()

    elif _query_strategy_name == "BatchBALD":
        _al_function = BatchBALD(num_samples=_query_batch_size)

    elif _query_strategy_name == "PowerBALD":
        _al_function = PowerSampling(BALD(), query_size=_query_batch_size, temperature=1.0)


    _temp_X_columns = [x for x, mask in zip(_df_train.columns.values, _df_train.columns.str.startswith("X")) if mask]

    X_train = _df_train[_df_train['sample_id'].isin(_cold_start_samples_id)].loc[:,_temp_X_columns].astype('float32')
    y_train = _df_train[_df_train['sample_id'].isin(_cold_start_samples_id)].loc[:,'labels'].astype('float32')	   

    X_test = _df_train.loc[:,_temp_X_columns].astype('float32')
    y_test = _df_train.loc[:,'labels'].astype('float32')

    X_validation = _df_validation.loc[:,_temp_X_columns].astype('float32')
    y_validation = _df_validation.loc[:,'labels'].astype('float32')			 

    #Cold Start Training:
    _model.fit(X_train, y_train)
    _array_score_train = cp.array(_model.score(X_test, y_test))
    _array_score_validation = cp.array(_model.score(X_validation, y_validation))

    #df Train Label & Unlabel:
    _array_labels_sample_ids = cp.array(_cold_start_samples_id)
    _array_unlabels_sample_ids = cp.array(_df_train['sample_id'][~_df_train['sample_id'].isin(_cold_start_samples_id)])    

    if _list_ordered_samples_id is not None:
        _list_ordered_samples_id = [x for x in _list_ordered_samples_id if x not in _cold_start_samples_id]
        _array_ordered_samples_id = cp.array(_list_ordered_samples_id)
        _temp_batch_size = 0


    _batch_looping = 1
    _array_batch_looping = cp.array(_batch_looping)

    _total_samples_evaluated = len(_cold_start_samples_id)
    _total_samples = len(_array_unlabels_sample_ids) + len(_array_labels_sample_ids)    
    _total_samples_evaluated_string = str(_total_samples_evaluated) + '/' + str(_total_samples)     
    _total_samples_evaluated_string_only = str(_total_samples_evaluated)
    # _array_total_samples_evaluated = cp.array(_total_samples_evaluated_string) #25/200
    _list_total_samples_evaluated = [_total_samples_evaluated_string] #25/200
    _list_total_samples_evaluated_only = [_total_samples_evaluated_string_only] #25
    
    _list_total_samples_evaluated_percetage = []
    _list_total_samples_evaluated_percetage.append(_total_samples_evaluated/_total_samples)

    # _array_samples_ids_per_batch = cp.array([_cold_start_samples_id])
    _list_samples_ids_per_batch = []
    _list_samples_ids_per_batch.append(_cold_start_samples_id)
        

    _list_time_query_selection = [0]
    _list_time_model_training = [0]
    _list_time_al_cycle = [0]




    #Here starts AL Cycle!
    while len(_array_unlabels_sample_ids) > 0:   

        start_time_al_cycle = time.time()       

        if len(_array_unlabels_sample_ids) % 50 == 0:
            print("Missing = ", len(_array_unlabels_sample_ids))
        

        start_time_query_selection = time.time()
        # If is Data-Density-Based:
        if _list_ordered_samples_id is not None:		
            #1) Get the most uncertainty -- this step was done before
            #2) Select Sample_ID based on the most uncertainty & query batch size
            selected_sample_id = _array_ordered_samples_id[_temp_batch_size:(_temp_batch_size+_query_batch_size)]
            _temp_batch_size = _temp_batch_size+_query_batch_size
            execution_time_query_selection = 0


        # If is Model-Based:
        else:
            
            #1) Predict in Ulabeled Train Dataset & Get the most uncertainty            
            #2) Select Sample_ID based on the most uncertainty & query batch size            
            _temp_test_x = _df_train[_temp_X_columns][_df_train['sample_id'].isin(_array_unlabels_sample_ids.get())]                     
            if _is_ensemble_model == True: 
                try:
                    #based on this tutorial: https://baal.readthedocs.io/en/latest/notebooks/compatibility/sklearn_tutorial/        
                    x = np.array(list(map(lambda e: e.predict_proba(_temp_test_x), _model.estimators_)))
                    x = np.rollaxis(x, 0, 3)
                    _baal_rank = _al_function(x)         
                    selected_sample_id = _array_unlabels_sample_ids[_baal_rank[:_query_batch_size]]            
                except AssertionError as e:
                    print(f"Shape of x before error: {x.shape}")
                    _list_total_samples_evaluated.append(None)
                    _list_total_samples_evaluated_only.append(None)
                    _list_total_samples_evaluated_percetage.append(None)
                    _list_samples_ids_per_batch.append(None)
                    _list_time_query_selection.append(None)
                    _list_time_model_training.append(None)
                    _list_time_al_cycle.append(None)                    
                    break                
            else:   
                try:
                    x = _model.predict_proba(_temp_test_x)
                    x = x.reshape(x.shape[0], x.shape[1], 1)		    
                    _baal_scores = _al_function.compute_score(x) 
                    _baal_rank = _al_function(x) 
                    #2) Select Sample_ID based on the most uncertainty & query batch size
                    selected_sample_id = _array_unlabels_sample_ids[_baal_rank[:_query_batch_size]]                        

                except AssertionError as e:
                    print(f"Shape of x before error: {x.shape}")
                    #raise e from None      
                    _list_total_samples_evaluated.append(None)
                    _list_total_samples_evaluated_only.append(None)
                    _list_total_samples_evaluated_percetage.append(None)
                    _list_samples_ids_per_batch.append(None)
                    _list_time_query_selection.append(None)
                    _list_time_model_training.append(None)
                    _list_time_al_cycle.append(None)                    
                    break

        end_time_query_selection = time.time()
        execution_time_query_selection = end_time_query_selection - start_time_query_selection

    	
 
        # 3) Remove selected sample_id from unlabels & add to labels        
        _array_unlabels_sample_ids = _array_unlabels_sample_ids[~cp.isin(_array_unlabels_sample_ids, selected_sample_id)]
        _array_labels_sample_ids= cp.append(_array_labels_sample_ids, selected_sample_id) 

        
        #4) Re-training the model
        start_time_model_training = time.time()
        X_train = _df_train[_df_train['sample_id'].isin(_array_labels_sample_ids.get())].loc[:,_temp_X_columns].astype('float32') 
        y_train = _df_train[_df_train['sample_id'].isin(_array_labels_sample_ids.get())].loc[:,'labels'].astype('float32')    
        _model.fit(X_train, y_train)                 
        end_time_model_training = time.time()
        execution_time_model_training = end_time_model_training - start_time_model_training            


        #5) Evaluate the model
        _score_train = cp.array(_model.score(X_test, y_test)) 
        _array_score_train = cp.append(_array_score_train, _score_train) 
	        
        _score_validation = cp.array(_model.score(X_validation, y_validation)) 
        _array_score_validation = cp.append(_array_score_validation, _score_validation) 


        #Dataframe inputs:
        _batch_looping = _batch_looping + 1
        _array_batch_looping = cp.append(_array_batch_looping, _batch_looping) 

        _total_samples_evaluated = _total_samples_evaluated + len(selected_sample_id)        
        _total_samples_evaluated_string = str(_total_samples_evaluated) + '/' + str(_total_samples)         
        _list_total_samples_evaluated.append(_total_samples_evaluated_string)        
        _list_total_samples_evaluated_only.append(_total_samples_evaluated_string_only)        
        _list_total_samples_evaluated_percetage.append(_total_samples_evaluated/_total_samples)        

        _list_samples_ids_per_batch.append(selected_sample_id.tolist())

        end_time_al_cycle = time.time()
        execution_time_al_cycle = end_time_al_cycle - start_time_al_cycle                  

        #TIME LISTS:
        _list_time_query_selection.append(execution_time_query_selection)        
        _list_time_model_training.append(execution_time_model_training)
        _list_time_al_cycle.append(execution_time_al_cycle)        



    _pd_list_0 = ['Framework_ID_' + str(_input_framework_id)] * len(_array_batch_looping)
    _pd_list_1 = [_database_name] * len(_array_batch_looping)
    _pd_list_2 = [_dl_architecture_name] * len(_array_batch_looping)
    _pd_list_3 = [_query_strategy_name] * len(_array_batch_looping)
    _pd_list_4 = [_model_name] * len(_array_batch_looping)
    _pd_list_5 = _array_batch_looping.tolist() #1, 2,... 10
    _pd_list_6 = _list_total_samples_evaluated #10/100, 20/100,... 100/100
    _pd_list_7 = _list_total_samples_evaluated_percetage #5%, 10%, ... 100%

    _pd_list_8 = _list_samples_ids_per_batch #[1,5,3,2,...], [...], [...]
    _pd_list_9 = _array_score_train.tolist() 
    _pd_list_10 = _array_score_validation.tolist() 

    _pd_list_11 = [_query_batch_size] * len(_array_batch_looping)
    _pd_list_12 = [_legend_name] * len(_array_batch_looping)

    _pd_list_13 = _list_total_samples_evaluated_only #10, 20,... 100

    _pd_list_14 = _list_time_model_training
    _pd_list_15 = _list_time_query_selection
    _pd_list_16 = _list_time_al_cycle



    _result_df = pd.DataFrame(list(zip(_pd_list_0, #Framework_ID                                            
                                            _pd_list_1, #Database", 
                                            _pd_list_2, #DL_Architecture", 
                                            _pd_list_3, #Query_Strategy", 
                                            _pd_list_12, #Query_Strategy_Batch (Legend),   
                                            _pd_list_11, #Batch Size",
                                            _pd_list_4, #Model", 
                                            _pd_list_5 ,  #Round",
                                           _pd_list_13,  #Samples Evaluated",
                                            _pd_list_6, #Samples Evaluated / Total Samples",
                                            _pd_list_7, #Percetage Samples Evaluated",
                                            _pd_list_8, #Samples IDs",
                                            _pd_list_9, #Samples Accuracy Train"
                                            _pd_list_10, #Samples Accuracy Validation"
                                            _pd_list_14, #Time Model Training (seconds)
                                            _pd_list_15, #Time Query Selection (seconds)
                                            _pd_list_16 #Time Model AL Cycle (seconds)
                                            ) 
                                        ) 
        ,columns=["Framework_ID", 
                  "Database", 
                  "DL_Architecture", 
                  "Query_Strategy", 
                  "Query_Strategy_Batch",   
                  "Batch Size",
                  "Model", 
                  "Round",
                  "Samples Evaluated",
                  "Samples Evaluated / Total Samples",
                  "Percetage Samples Evaluated",
                  "Samples IDs",
                  "Samples Accuracy Train",
                  "Samples Accuracy Validation",
                  "Time Model Training (seconds)",
                  "Time Query Selection (seconds)",
                  "Time Model AL Cycle (seconds)"
                  ]                
    )							

    return _result_df











for db_paths in _list_data_sets_path:



    _sub_folders_to_check = f_get_subfolders(db_paths[0])
    for _sub_folder in _sub_folders_to_check:	
        f_delete_files(f_get_files_to_delete(_script_name), _sub_folder)		

    
    _deep_learning_arq_sub_folders =  [db_paths for db_paths in os.listdir(db_paths[4]) if not db_paths.startswith('.')]
    for _deep_learning_arq_sub_folder_name in _deep_learning_arq_sub_folders: #vgg_16, #vgg_19,... 									


        #Open Files!         
        _df_train = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_train.pkl')
        _df_faiss_indices = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_faiss_indices_train.pkl')
        _df_faiss_distances = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_faiss_distances_train.pkl')
        _df_validation = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_validation.pkl')
        
        _df_2D_train = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_2D_train.pkl')
        _df_2D_faiss_indices = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_2D_faiss_indices_train.pkl')
        _df_2D_faiss_distances = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_2D_faiss_distances_train.pkl')
        _df_2D_validation = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_2D_validation.pkl')

        print("\n\n\n\n\n\n")
        print("OPEN PATH === ", db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'label_encoder.pkl')
                   
        _label_encoder = pickle.load(open(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'label_encoder.pkl', 'rb'))             
        _df_train['labels'] = _label_encoder.transform(_df_train['labels'])
        _df_validation['labels'] = _label_encoder.transform(_df_validation['labels'])            

        print("classes = ", _label_encoder.classes_)
        print("\n\n\n\n\n\n") 


        #########################################################################################################             
        ######### ORDERED SAMPLES CALCULATION
        #########################################################################################################             

        _random_samples_id, _cold_start_samples_id = bblocks.f_cold_start(_df_train)       

        _dict_simumlations_ordered_samples = {                
            'Random': _random_samples_id,
            'Uncertainty': None, 
            'Margin': None, 
            'Entropy': None, 
            'Bald': None, 
            'BatchBALD': None, 
            'PowerBALD': None
        }       

        if config.human_simulations == True:        
            if config.load_human_simulations_files == True:
                _simulation_order_df = pd.read_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_simulation_order_df.pkl')
                _simulation_order_df_2D = pd.read_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_simulation_order_df_2D.pkl')

            else:
                _list_strategy_name, _list_strategy_ordered_samples_id = bblocks.f_run_human_simulations(df_embbedings = _df_train, 
                                                        df_faiss_indices=_df_faiss_indices, 
                                                        df_faiss_distances=_df_faiss_distances)
                _simulation_order_df = pd.DataFrame(_list_strategy_ordered_samples_id).T
                _simulation_order_df.columns = _list_strategy_name	
                _simulation_order_df.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_simulation_order_df.pkl')


                _list_strategy_name_2D, _list_strategy_ordered_samples_id_2D = bblocks.f_run_human_simulations(df_embbedings = _df_2D_train, 
                                                        df_faiss_indices=_df_2D_faiss_indices, 
                                                        df_faiss_distances=_df_2D_faiss_distances)
                _simulation_order_df_2D = pd.DataFrame(_list_strategy_ordered_samples_id_2D).T
                _simulation_order_df_2D.columns = _list_strategy_name_2D	 
                _simulation_order_df_2D.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_simulation_order_df_2D.pkl')


            #Update the Dictionary with Ordered Samples
            new_samples_lists = [
                list(_simulation_order_df['Equal_Spread'].values),
                list(_simulation_order_df['Dense_Areas_First'].values),
                list(_simulation_order_df['Centroids_First'].values),
                list(_simulation_order_df['Outliers_First'].values),

                list(_simulation_order_df_2D['Equal_Spread'].values),
                list(_simulation_order_df_2D['Dense_Areas_First'].values),
                list(_simulation_order_df_2D['Centroids_First'].values),
                list(_simulation_order_df_2D['Outliers_First'].values)                    
            ]
            new_keys = ['Equal_Spread', 'Dense_Areas_First', 'Centroids_First', 'Outliers_First',
                        'Equal_Spread_2D', 'Dense_Areas_First_2D', 'Centroids_First_2D', 'Outliers_First_2D']
        
        

            for key, value in zip(new_keys, new_samples_lists):
                _dict_simumlations_ordered_samples[key] = value


        ######################################################################################################### 
        #########################################################################################################             
        ######### ACTIVE LEARNING CYCLE!            
        ######################################################################################################### 
        ######################################################################################################### 


        _list_dfs = []
        _list_query_stragegy = config._list_query_stragegy
        _batch_size_options = config._batch_size_options
                           

        for i in range(len(_list_query_stragegy)):

            if _list_query_stragegy[i] in config._list_strategies_for_batch_size_comparison:                     
                if _batch_size_experiment == True:
                    _batch_size_options = config._batch_size_options
                    _batch_size_options.append(int(round(_df_train.shape[0]/25,0)))

                else:
                    _batch_size_options = [int(round(_df_train.shape[0]/25,0))]


                for _b_size in _batch_size_options:


                    _df_temp = f_framework_df(
                        _df_train = _df_train, 
                        _df_validation = _df_validation, 
                        _cold_start_samples_id = _cold_start_samples_id, 
                        _legend_name = _list_query_stragegy[i] + '_batch_' + str(_b_size),
                        _query_strategy_name = _list_query_stragegy[i],
                        _query_batch_size = _b_size,
                        _database_name = db_paths[0].split('/')[1],
                        _dl_architecture_name = _deep_learning_arq_sub_folder_name, 
                        # _df_faiss_indices=_df_faiss_indices,
                        # _df_faiss_distances=_df_faiss_distances,
                        _list_ordered_samples_id=_dict_simumlations_ordered_samples[_list_query_stragegy[i]],
                        _input_framework_id = i+1                            
                        )
                    _list_dfs.append(_df_temp)                       

            else:                

                _df_temp = f_framework_df(
                    _df_train = _df_train, 
                    _df_validation = _df_validation, 
                    _cold_start_samples_id = _cold_start_samples_id, 
                    _legend_name = _list_query_stragegy[i] + '_batch_' + str(_b_size),
                    _query_strategy_name = _list_query_stragegy[i],
                    _query_batch_size = int(round(_df_train.shape[0]/25,0)),
                    _database_name = db_paths[0].split('/')[1],
                    _dl_architecture_name = _deep_learning_arq_sub_folder_name, 
                    # _df_faiss_indices=_df_faiss_indices,
                    # _df_faiss_distances=_df_faiss_distances,
                    _list_ordered_samples_id=_dict_simumlations_ordered_samples[_list_query_stragegy[i]],
                    _input_framework_id = i+1                        
                    )
                _list_dfs.append(_df_temp)

            #Save a temporary Pickle 
            try:                    
                _df_temp_del = pd.read_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_framework_temporary.pkl')    
                _df_temp = _df_temp_del.append(_df_temp)
            except:
                None

            _df_temp.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_framework_temporary.pkl')



        df_final = pd.concat(_list_dfs)
        df_final = df_final.reset_index(drop=True)
        df_final.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_framework.pkl')