import os
import sys
import warnings
from warnings import filterwarnings
import random
import time
import datetime

import pandas as pd
import numpy as np
import seaborn as sns
from imutils import paths

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning

from tqdm import tqdm
import concurrent.futures
import multiprocessing
from joblib import Parallel, delayed

# import cudf
# import cuml

import aux_functions
import config as config
config = config.config

#Inputs:
_GPU_flag = config._GPU_Flag_dict['06_framework.py']

_list_data_sets_path = config._list_data_sets_path
_list_train_val = config._list_train_val

warnings.filterwarnings("ignore", category=DeprecationWarning)
filterwarnings('ignore')




with open(f_time_now(_type='datetime_') + 'logs/06_framework_py_' + ".txt", "a") as f:
    _string_log_input = ['[INFO] Starting Simulation Framework', 0]    
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


                        _list_databases_training = [] # list
                        _list_databases_test = [] # list
                        _list_databases_name = [] # list                    
                        _list_simulation_ordered_samples_id = [] #list of list
                        
                        _results_output = [                        
                            [], # 0 - Outcome Interaction                        
                            [], # 1 - Name                        
                            [], # 2 - Dataset                        
                            [], # 3 - DL Arq                        
                            [], # 4 - Simulation                        
                            [], # 5 - Model Name                        
                            [], # 6 - Model Parameters                        
                            [], # 7 - List w/ Total of Labels Evaluated:                        
                            [], # 8 - List w/ Accuracy for each sum of Labels:
                        ]

                        

                        if _GPU_flag is True:
                            df = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + _file_name)                        
                            df = cudf.DataFrame.from_pandas(df)

                            _string_log_input = ['[HILIGHT] as GPU_Flag = True, using cuDF Library for optmization' + _file_name, 5]    
                            f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
                            f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))

                        else:
                            df = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + _file_name)

                        
                        if _GPU_flag is True:
                            _list_models = [cuml.LogisticRegression()]  # list
                        else:
                            _list_models = [LogisticRegression(random_state=0)]  # list
                        

                        _list_models_name = []
                        for i in range(len(_list_models)): 
                            _list_models_name.append(type(_list_models[0]).__name__)

                    
                        if _GPU_flag is True:
                            df_faiss_indices = pd.read_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_faiss_indices_' + _list_train_val[i_train_val]  + '.pkl')
                            df_faiss_distances = pd.read_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_faiss_distances_' + _list_train_val[i_train_val]  + '.pkl')     
                            df_faiss_indices = cudf.DataFrame.from_pandas(df_faiss_indices)
                            df_faiss_distances = cudf.DataFrame.from_pandas(df_faiss_distances)

                            _string_log_input = ['[HILIGHT] as GPU_Flag = True, using cuDF Library for optmization' + _file_name, 5]    
                            f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
                            f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))                            

                            
                            
                        else:
                            df_faiss_indices = pd.read_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_faiss_indices_' + _list_train_val[i_train_val]  + '.pkl')
                            df_faiss_distances = pd.read_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_faiss_distances_' + _list_train_val[i_train_val]  + '.pkl')                             
                            

                        _string_log_input = ['[INFO] Importing Simulations .pkl...' + _file_name, 5]    
                        f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
                        f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))                            


                        # _list_simulation_sample_name, _list_simulation_ordered_samples_id = sim.f_run_simulations(df_embbedings = df, df_faiss_indices=df_faiss_indices, df_faiss_distances=df_faiss_distances, simulation_list = None)
                        _df_simulation_ordered = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder_name + '/' + 'df_simulation_ordered_' + _list_train_val[i_train_val]  + '.pkl')
                        _list_simulation_sample_name = list(_df_simulation_ordered.columns)                    
                        _list_simulation_ordered_samples_id = _df_simulation_ordered.T.values.tolist()                                       
                        _df_simulation_ordered = None
                        

                        _string_log_input = ['[INFO] Starting ML Framework.py....' + _file_name, 5]    
                        f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
                        f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))  


                        i_Outcome_ = 1
                        for i_model in range(len(_list_models)):                                  
                            _string_log_input = ['Model = ' +  _list_models_name[i_model] + _file_name, 6]    
                            f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
                            f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))  

                            start_time_model = time.time()
                            for i_simulation in range(len(_list_simulation_sample_name)):

                                _string_log_input = ['Simulation = ' +  _list_simulation_sample_name[i_simulation], 7]    
                                f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
                                f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))

                                _list_accuracy_on_labels_evaluated = []
                                _list_labels_evaluated = np.arange(1, df.shape[0] + 1, 1).tolist()

                                

                                # print (db_paths[0].split('/')[1], " | ", _deep_learning_arq_sub_folder_name , '| ', _list_simulation_sample_name[i_simulation], " | ", _list_models_name[i_model])
                                _ordered_samples_id = _list_simulation_ordered_samples_id[i_simulation]

                                _string_log_input = ['Qtd. Samples To Run = ' +  '{:,.0f}'.format(len(_ordered_samples_id)).replace(',', ';').replace('.', ',').replace(';', '.'), 8]    
                                f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
                                f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))

                                _model = _list_models[i_model]                               
                            
                                start_time_simulation = time.time()                                
                                chunk_size = round(len(_ordered_samples_id) / 250)
                                # with tqdm(total=len(_ordered_samples_id)) as pbar:
                                # pbar.update(chunk_size)                           

                                for i in range(0, len(_ordered_samples_id), chunk_size):                                
                                    # chunk = _ordered_samples_id[i:i+chunk_size]                                

                                    list1 = [df.copy(deep=True) for _ in range(chunk_size)]
                                    list2 = [_model for _ in range(chunk_size)]
                                    list3 = [_ordered_samples_id for _ in range(chunk_size)]
                                    list4 = list(range(i, (i+chunk_size)))
                                    list5 = [_GPU_flag for _ in range(chunk_size)]

                                    #list_of_lists_f_model_accuracy = [list1, list2, list3, list4, list5]
                                    tuple_f_model_accuracy = [(a, b, c, d, e) for a, b, c, d, e in zip(list1, list2, list3, list4, list5)]
                                                                                
                                    #[TO-DO] Create a function and parallelize with Multithread --> "Done, check if is ok"
                                    results = Parallel(n_jobs=num_cores)(delayed(f_model_accuracy)(args) for args in tuple_f_model_accuracy)                                                                    
                                    _list_accuracy_on_labels_evaluated.append(results)                                                
                                

                                end_time_simulation = time.time()                            
                                time_taken_simulation = (end_time_simulation - start_time_simulation)
                                

                                _string_log_input = ['Time Taken Simulation: {:.2f} minutes'.format(time_taken_simulation), 8]    
                                f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
                                f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))

                
                                _name_temp = db_paths[0].split('/')[1] + " | " + _deep_learning_arq_sub_folder_name  + '| ' + _list_simulation_sample_name[i_simulation] + " | " + _list_models_name[i_model]                            
                                _results_output[0].append('Outcome_' + str(i_Outcome_))
                                _results_output[1].append(_name_temp)
                                _results_output[2].append(db_paths[0].split('/')[1])
                                _results_output[3].append(_deep_learning_arq_sub_folder_name)                            
                                _results_output[4].append(_list_simulation_sample_name[i_simulation])            
                                _results_output[5].append(_list_models_name[i_model])
                                _results_output[6].append(_model.get_params())            
                                _results_output[7].append(_list_labels_evaluated)
                                _results_output[8].append([num for sublist in _list_accuracy_on_labels_evaluated for num in sublist])                                    
                                i_Outcome_ = i_Outcome_ + 1

                            end_time_model = time.time()                            
                            time_taken_model = (end_time_model - start_time_model)

                            _string_log_input = ['Time Taken Model: {:.2f} minutes'.format(time_taken_simulation), 7]    
                            f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
                            f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))


                        _string_log_input = ['[INFO] Simulation Results - DataFrame Creation', 5]    
                        f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
                        f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))
                        
                        _temp_df_list = []
                        #[TO-DO] Parallelize this looping with multithread
                        for _i_outcome in range(len(_results_output[0])):
                            
                            _number_of_samples = len(_results_output[7][_i_outcome])                        
                            _pd_list_0 = [_results_output[0][_i_outcome]] * _number_of_samples
                            _pd_list_1 = [_results_output[1][_i_outcome]] * _number_of_samples
                            _pd_list_2 = [_results_output[2][_i_outcome]] * _number_of_samples                        
                            _pd_list_3 = [_results_output[3][_i_outcome]] * _number_of_samples
                            _pd_list_4 = [_results_output[4][_i_outcome]] * _number_of_samples
                            _pd_list_5 = [_results_output[5][_i_outcome]] * _number_of_samples
                            #_pd_list_6 = _results_output[6][_i_outcome] * _number_of_samples
                            _pd_list_7 = _results_output[7][_i_outcome] 
                            _pd_list_8 = _results_output[8][_i_outcome] 
                            
                            #[TO-DO] Create a cuDF
                            _temp_df = pd.DataFrame(
                                list(zip(_pd_list_0, _pd_list_1, _pd_list_2, _pd_list_3, _pd_list_4, _pd_list_5 ,_pd_list_7,_pd_list_8))
                                ,columns=["Outcome", "Interaction", "Database", "DL Architecture", "Simulation Type", "Model", "# Samples Evaluated/Interaction Number","Accuracy"]
                                # ,_pd_list_6 ,"Model Parameters"
                            )                                                        
                            _temp_df_list.append(_temp_df)

                        #[TO-DO] Create a cuDF and transform to pickle
                        df_simulation = pd.concat(_temp_df_list)
                        df_simulation.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'df_simulation_' + _list_train_val[i_train_val] + '.pkl')
                        
                        

                        _string_log_input = ['[INFO] Chart Creation', 5]    
                        f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=False)        
                        f_write(f_print(_string_log_input[0], _level=_string_log_input[1], _write_option=True))                        


                        #[TO-DO] Create a function to generate the chart
                        f_create_chart(df_simulation, 
                                        _path=db_paths[4] +'/' + _deep_learning_arq_sub_folder_name + '/' + 'chart' + _list_train_val[i_train_val] + '.png')