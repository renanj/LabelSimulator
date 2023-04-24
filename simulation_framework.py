from shelve import DbfilenameShelf
from sqlite3 import DatabaseError
from tkinter.ttk import LabeledScale
import simulation as sim
import pandas as pd
import numpy as np
import seaborn as sns
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imutils import paths
import os
from tqdm import tqdm

import time

import concurrent.futures
from joblib import Parallel, delayed
import multiprocessing

# import cudf
# import cuml


import config as config
config = config.config



num_cores = multiprocessing.cpu_count()
print("[INFO] num_cores = ", num_cores)


_GPU_flag = False

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
        _temp_X_columns = list(_df.loc[:,_df.columns.str.startswith("X")].columns)                                                                
        X_train = _df[_df['sample_id'].isin(_ordered_samples_id_temp)].loc[:,_temp_X_columns]
        y_train = _df[_df['sample_id'].isin(_ordered_samples_id_temp)].loc[:,'labels']                      
        X_test = df.loc[:,_temp_X_columns]
        y_test = df.loc[:,'labels']


    try:                                    
        _model.fit(X_train, y_train)                                    
        _score = _model.score(X_test, y_test)
        print("worked for..", _qtd_samples_to_train)
        return _score
        
    except:                                            
        _score = 0
        print("entered i expection...")
        return _score



#Plancton, mnist, etc...
print('[INFO] Starting Simulation Framework')
for db_paths in config._list_data_sets_path:
    print(db_paths[0])
    _deep_learning_arq_sub_folders =  [db_paths for db_paths in os.listdir(db_paths[4]) if not db_paths.startswith('.')]
    for _deep_learning_arq_sub_folders in _deep_learning_arq_sub_folders:
        print('-------')
        print('.../' + _deep_learning_arq_sub_folders)        
        _list_files = [_files for _files in os.listdir(db_paths[4] + '/' + _deep_learning_arq_sub_folders) if not _files.startswith('.')]        
        print("List of Files: ", _list_files)
        

        for i_train_val in range(len(config._list_train_val)):            
            for _files in _list_files:
                print(_files)
                if _files !='df_'+ config._list_train_val[i_train_val] + '.pkl':
                    None
                else:                    
                    print ("Running:  ", _files)                    

            #Here Starts the Simulation for Each DB        
            #----------------------------------------                                                             
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

                    #[TO-DO] change this to be dynamic... 
                    _list_simulation_sample_pallete = ['#F22B00', '#40498e', '#357ba3', '#38aaac', '#79d6ae']
                    
                    #[TO-DO] transform to cuDF
                    if _GPU_flag is True:
                        df = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folders + '/' + _files)                        
                        df = cudf.DataFrame.from_pandas(df)
                        print("[HILIGHT] as GPU_Flag = True, using cuDF Library for optmization")
                    else:
                        df = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folders + '/' + _files)

                    #[TO-DO] transform to cuML -- Done, check if is ok
                    if _GPU_flag is True:
                        _list_models = [cuml.LogisticRegression()]  # list
                    else:
                        _list_models = [LogisticRegression(random_state=0)]  # list
                    

                    _list_models_name = []
                    for i in range(len(_list_models)): 
                        _list_models_name.append(type(_list_models[0]).__name__)

    
                    print("[INFO] Starting simulation.py...")
                    _list_simulation_sample_name, _list_simulation_ordered_samples_id = sim.f_run_simulations(df_embbedings = df, simulation_list = None)
                    
                    print("[INFO] Starting ML Framework") 
                    i_Outcome_ = 1
                    for i_model in range(len(_list_models)):                               
                        for i_simulation in range(len(_list_simulation_ordered_samples_id)):

                            _list_accuracy_on_labels_evaluated = []
                            _list_labels_evaluated = np.arange(1, df.shape[0] + 1, 1).tolist()

                            print (db_paths[0].split('/')[1], " | ", _deep_learning_arq_sub_folders , '| ', _list_simulation_sample_name[i_simulation], " | ", _list_models_name[i_model])                                                    
                            _ordered_samples_id = _list_simulation_ordered_samples_id[i_simulation]
                            _model = _list_models[i_model]                               
                        
                            start_time = time.time()

                            list1 = [df.copy(deep=True) for _ in range(len(_ordered_samples_id))]
                            list2 = [_model for _ in range(len(_ordered_samples_id))]
                            list3 = [_ordered_samples_id for _ in range(len(_ordered_samples_id))]  
                            list4 = list(range(0, len(_ordered_samples_id)))
                            list5 = [_GPU_flag for _ in range(len(_ordered_samples_id))]

                            #list_of_lists_f_model_accuracy = [list1, list2, list3, list4, list5]
                            tuple_f_model_accuracy = [(a, b, c, d, e) for a, b, c, d, e in zip(list1, list2, list3, list4, list5)]
                                                                            
                            #[TO-DO] Create a function and parallelize with Multithread --> "Done, check if is ok"
                            results = Parallel(n_jobs=num_cores)(delayed(f_model_accuracy_5)(args) for args in tuple_f_model_accuracy)
                            _list_accuracy_on_labels_evaluated = list(results)

                            end_time = time.time()                            
                            time_taken = (end_time - start_time)/60
                            print("Time taken: {:.2f} minutes".format(time_taken))

            
                            _name_temp = db_paths[0].split('/')[1] + " | " + _deep_learning_arq_sub_folders  + '| ' + _list_simulation_sample_name[i_simulation] + " | " + _list_models_name[i_model]                            
                            _results_output[0].append('Outcome_' + str(i_Outcome_))
                            _results_output[1].append(_name_temp)
                            _results_output[2].append(db_paths[0].split('/')[1])
                            _results_output[3].append(_deep_learning_arq_sub_folders)                            
                            _results_output[4].append(_list_simulation_sample_name[i_simulation])            
                            _results_output[5].append(_list_models_name[i_model])
                            _results_output[6].append(_model.get_params())            
                            _results_output[7].append(_list_labels_evaluated)
                            _results_output[8].append(_list_accuracy_on_labels_evaluated)                                    
                            i_Outcome_ = i_Outcome_ + 1

                    
                    print("[INFO] Results DataFrame Creation") 
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
                            ,columns=["Outcome", "Interaction", "Database", "DL Architecture", "Simulation Type", "Model", "# Samples Evaluated/Interaction Number", ,"Accuracy"]
                            # ,_pd_list_6 ,"Model Parameters"
                        )                                                        
                        _temp_df_list.append(_temp_df)

                    #[TO-DO] Create a cuDF and transform to pickle
                    df_simulation = pd.concat(_temp_df_list)
                    df_simulation.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folders + '/' + 'df_simulation_' + config._list_train_val[i_train_val] + '.pkl')
                    

                    print("[INFO] Chart Creation") 
                    #[TO-DO] Create a function to generate the chart
                    _temp_df_chart = df_simulation[['Simulation Type', '# Samples Evaluated/Interaction Number', 'Accuracy']]
                    _temp_df_chart = _temp_df_chart.reset_index(drop=True)                    
                    palette = sns.color_palette("mako", len(_list_models))
                    
                    sns.set(rc={'figure.figsize':(15.7,8.27)})

                    palette= _list_simulation_sample_pallete

                    _chart = None
                    _chart = sns.lineplot(data=_temp_df_chart, 
                                x="# Samples Evaluated/Interaction Number", 
                                y="Accuracy", 
                                hue="Simulation Type",
                                palette=palette
                                )

                    figure = _chart.get_figure()
                    figure.savefig(db_paths[4] +'/' + _deep_learning_arq_sub_folders + '/' + 'chart' + config._list_train_val[i_train_val] + '.png')                                        