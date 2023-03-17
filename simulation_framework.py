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

import config as config
config = config.config



#Plancton, mnist, etc...
print('[INFO] Starting Simulation Framework')
for db_paths in config._list_data_sets_path:
    print("\n\nPATH -------")
    print('=====================')
    print(db_paths[0])
    print('=====================')
    print('=====================\n')
    #folders for db extract (vgg16, vgg18, etc)
    _deep_learning_arq_sub_folders =  [db_paths for db_paths in os.listdir(db_paths[4]) if not db_paths.startswith('.')]    
    for _deep_learning_arq_sub_folders in _deep_learning_arq_sub_folders:
        print('-------')
        print('.../' + _deep_learning_arq_sub_folders)
        #list of files
        _list_files = [_files for _files in os.listdir(db_paths[4] + '/' + _deep_learning_arq_sub_folders) if not _files.startswith('.')]
        print("LIST_FILES --->")
        print(_list_files)
        #split in train & validation (currently we use only validation)        
        for i_train_val in range(len(config._list_train_val)):
            #print('... /...', config._list_train_val[i])
            for _files in _list_files:
                print(_files)
                if _files !='df_'+ config._list_train_val[i_train_val] + '.pkl':
                    None
                else:
                    print ('    ' + _files)
                    #print('.../' + _deep_learning_arq_sub_folder + '/' + _files)
                    
                    
            #Here Starts the Simulation for Each DB        
            #++++++ ++++++ ++++++ ++++++ ++++++ ++++++ ++++++ ++++++ ++++++ ++++++ ++++++                    
                     
                    df = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folders + '/' + _files)


                    # Database
                    _list_databases_training = [] # list
                    _list_databases_test = [] # list
                    _list_databases_name = [] # list
                    #Simulation
                    _list_simulation_samples = [] #list of list

                    # INPUT -- INPUT -- INPUT -- INPUT -- INPUT -- INPUT -- INPUT -- INPUT -- 
                    # MODELS: 
                    _list_models = [
                        LogisticRegression(random_state=0)
                    ]  # list

                    #models name -- based on sklearn
                    _list_models_name = []
                    for i in range(len(_list_models)): 
                        _list_models_name.append(type(_list_models[0]).__name__)


                    # RESULTS
                    _results_output = [
                        # Outcome Interaction
                        [],
                        # Name
                        [],    
                        #"Dataset
                        [],
                        #DL Arq
                        [],
                        # Simulation
                        [],
                        #Model Name
                        [],
                        #Model Parameters
                        [],
                        #List w/ Total of Labels Evaluated:
                        [],
                        #List w/ Accuracy for each sum of Labels:
                        [],
                    ]


                    # _path_training = 'data/plancton/splited/train/plancton_train.pkl'
                    # _path_test = 'data/plancton/splited/val/plancton_val.pkl'

                    # df_training = pd.read_pickle(_path_training)
                    # df_test = pd.read_pickle(_path_test)

                    # _list_databases_training.append(df_training)
                    # _list_databases_test.append(df_test)
                    # _list_databases_name.append('plancton')



                    # INPUT -- INPUT -- INPUT -- INPUT -- INPUT -- INPUT -- INPUT -- INPUT -- 
                    _list_simulation_sample_name = ['Random', 'NSS', 'SPB','DEN', 'OUT']
                    _list_simulation_sample_pallete = ['#F22B00', '#40498e', '#357ba3', '#38aaac', '#79d6ae']



                    #_list_simulation_samples.append([])
                    
                    #Random:    
                    _list_simulation_samples.append(random.sample(
                        range(df.shape[0]), 
                        df.shape[0]))
                    

                    #NSS:
                    _temp_list = sim.f_NSS(df)
                    _list_simulation_samples.append(_temp_list)

                    #SPB:
                    _temp_list = sim.f_NSS(df)
                    _list_simulation_samples.append(_temp_list)

                    #DEN:
                    _temp_list = sim.f_DEN(df)
                    _list_simulation_samples.append(_temp_list)    

                    #OUT:
                    _temp_list = sim.f_OUT(df)
                    _list_simulation_samples.append(_temp_list)    







                    # ML FRAMEWORK
                    i_Outcome_ = 1
                    for i_model in range(len(_list_models)):       
                        #Baseado no Framework de Simulation Samples
                        for i_simulation in range(len(_list_simulation_samples)):
                            
                            print (db_paths[0].split('/')[1], " | ", _deep_learning_arq_sub_folders , '| ', _list_simulation_sample_name[i_simulation], " | ", _list_models_name[i_model])

                            
                            # _db = _list_databases_training[i_db]
                            # _db_test = _list_databases_test[i_db]
                            #[TO-DO]: tem que checar se o dataset de training esta alinhadoi com o datasert de test…se nao vai dar merda

                            _samples = _list_simulation_samples[i_simulation]
                            _model = _list_models[i_model]

                            #[TO-DO]: fazre um teste se o dataset tem X1, X2... etc
                            _temp_X_columns = list(df.loc[:,df.columns.str.startswith("X")].columns)
                    
                            # print("db = ", i_db)
                            # print("simulation = ", i_simulation)
                            # print("model = ", i_model)
                            # print("------------------\n")
                            

                            _list_accuracy_on_labels_evaluated = []
                            LOG_EVERY_N = 1000
                            #Framework de Accuracy
                            for i in range(len(_samples)):

                                if ((i + 1) % LOG_EVERY_N) == 0:
                                    print ("Interaction = ", i + 1 / len(samples))

                                _samples_temp = _samples[0:i+1]

                                #prepara X and y
                                X_train = df[df['sample_id'].isin(_samples_temp)].loc[:,_temp_X_columns]
                                y_train_true = df[df['sample_id'].isin(_samples_temp)].loc[:,'labels']

                                
                                try:
                                    #FIT - train model:
                                    _model.fit(X_train, y_train_true)

                                    #db test (evaluation):
                                    #Evaluation on TEST (another df)
                                    #X_test = _db_test.loc[:,_temp_X_columns]
                                    #y_test_true = _db_test.loc[:,'labels']


                                    #Evaluation on SAME DF using everything
                                    X_test = df.loc[:,_temp_X_columns]
                                    y_test_true = df.loc[:,'labels']

                                    #Predict:
                                    #y_test_predict = _model.predict_proba()

                                    #Accuracy Results 
                                    #_list_accuracy_on_labels_evaluated = accuracy_score(y_test_true, y_test_predict)
                                    _list_accuracy_on_labels_evaluated.append(_model.score(X_test, y_test_true))
                                        # LIS OT ACCURACY!!!!!!!
                                except:
                                    _list_accuracy_on_labels_evaluated.append(0)
                                    #print("Not entered in ML Fit")

                            _list_labels_evaluated = np.arange(1, df.shape[0] + 1, 1).tolist()
                            # LITS -- outcome FROM FRAMEWORK OF ACCURACY!!!!!:
                            # _list_labels_evaluated = np.arange(1, 301, 1).tolist()
                            # _list_accuracy_on_labels_evaluated = random.sample(range(15), 15)
                            
                    
                            #RESULTS OUTPUT:
                            #name:                            
                            #_name_temp = db_paths[0].split('/')[1] + ' | ' + _list_simulation_sample_name[i_simulation] + ' | ' + _list_models_name[i_model]
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


                    # Criar Pandas Dataset para depois gerar grafico!
                    _temp_df_list = []
                    for _i_outcome in range(len(_results_output[0])):
                        
                        _number_of_samples = len(_results_output[6][_i_outcome])
                        
                        _pd_list_0 = [_results_output[0][_i_outcome]] * _number_of_samples
                        _pd_list_1 = [_results_output[1][_i_outcome]] * _number_of_samples
                        _pd_list_2 = [_results_output[2][_i_outcome]] * _number_of_samples                        
                        _pd_list_3 = [_results_output[3][_i_outcome]] * _number_of_samples
                        _pd_list_4 = [_results_output[4][_i_outcome]] * _number_of_samples
                        _pd_list_5 = [_results_output[5][_i_outcome]] * _number_of_samples
                        #_pd_list_6 = _results_output[6][_i_outcome] * _number_of_samples
                        _pd_list_7 = _results_output[7][_i_outcome] 
                        _pd_list_8 = _results_output[8][_i_outcome] 
                        
                            
                        _temp_df = pd.DataFrame(list(
                            zip(
                                _pd_list_0
                                ,_pd_list_1
                                ,_pd_list_2
                                ,_pd_list_3
                                ,_pd_list_4
                                ,_pd_list_5                                
                                # ,_pd_list_6
                                ,_pd_list_7
                                ,_pd_list_8                                            
                                ))
                            ,columns=[
                                "Outcome"
                                ,"Interaction"
                                ,"Database"
                                ,"DL Architecture"
                                ,"Simulation Type"
                                ,"Model"
                                #,"Model Parameters"
                                ,"# Samples Evaluated/Interaction Number"
                                ,"Accuracy"
                            ]
                        )
                        
                        _temp_df_list.append(_temp_df)

                    df_simulation = pd.concat(_temp_df_list)                
                    df_simulation.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folders + '/' + 'df_simulation.pkl')

                    #Chart - Chart - Chart:
                    _temp_df_chart = df_simulation[['Simulation Type', '# Samples Evaluated/Interaction Number', 'Accuracy']]
                    _temp_df_chart = _temp_df_chart.reset_index(drop=True)                    
                    palette = sns.color_palette("mako", len(_list_models))
                    
                    sns.set(rc={'figure.figsize':(15.7,8.27)})

                    palette= _list_simulation_sample_pallete


                    _chart = sns.lineplot(data=_temp_df_chart, 
                                x="# Samples Evaluated/Interaction Number", 
                                y="Accuracy", 
                                hue="Simulation Type",
                                palette=palette
                                )

                    figure = _chart.get_figure()
                    figure.savefig(db_paths[4] +'/' + _deep_learning_arq_sub_folders + '/' + 'chart.png')                                        