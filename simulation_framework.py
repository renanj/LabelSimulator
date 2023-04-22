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


import config as config
config = config.config


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
                    _list_simulation_samples = [] #list of list
                    
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
                    df = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folders + '/' + _files)                    
                    #[TO-DO] transform to cuML:
                    _list_models = [LogisticRegression(random_state=0)]  # list                    
                    _list_models_name = []
                    for i in range(len(_list_models)): 
                        _list_models_name.append(type(_list_models[0]).__name__)

    
                    print("[INFO] Starting simulation.py...")
                    _list_simulation_sample_name, _list_simulation_samples = sim.f_run_simulations(df_embbedings = df, simulation_list = None)
                    
                    print("[INFO] Starting ML Framework") 

                    i_Outcome_ = 1
                    for i_model in range(len(_list_models)):                               
                        for i_simulation in range(len(_list_simulation_samples)):


                            _list_accuracy_on_labels_evaluated = []
                            _list_labels_evaluated = np.arange(1, df.shape[0] + 1, 1).tolist()


                            print (db_paths[0].split('/')[1], " | ", _deep_learning_arq_sub_folders , '| ', _list_simulation_sample_name[i_simulation], " | ", _list_models_name[i_model])                                                    
                            _samples = _list_simulation_samples[i_simulation]
                            _model = _list_models[i_model]   

                            _temp_X_columns = list(df.loc[:,df.columns.str.startswith("X")].columns)                
                            
                            #[TO-DO] Create a function and parallelize with Multithread
                            for i in tqdm(range(len(_samples))):

                                _samples_temp = _samples[0:i+1]
                                #[TO-DO] use cudf... or give an option when using GPU vs. CPU
                                X_train = df[df['sample_id'].isin(_samples_temp)].loc[:,_temp_X_columns]
                                y_train_true = df[df['sample_id'].isin(_samples_temp)].loc[:,'labels']
                                
                                try:                                    
                                    _model.fit(X_train, y_train_true)
                                    X_test = df.loc[:,_temp_X_columns]
                                    y_test_true = df.loc[:,'labels']
                                    _list_accuracy_on_labels_evaluated.append(_model.score(X_test, y_test_true))
                                        
                                except:
                                    _list_accuracy_on_labels_evaluated.append(0)
                                
                    
                                                    
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