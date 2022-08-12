from shelve import DbfilenameShelf
from sqlite3 import DatabaseError
from tkinter.ttk import LabeledScale
import simulation as sim
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



#Names
# _list_databases_name = ['database1']

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
    #Name as "Dataset Name | Simulation Name | Model Name"
    [],
    #List w/ Total of Labels Evaluated:
    [],
    #List w/ Accuracy for each sum of Labels:
    [],
]


# DATABASE + SAMPLE:
_error = False
i = 1
while _error == False:

    try:
        _path_training = 'data/dataset_' + str(i) + '/processed/training/df.csv'
        _path_test = 'data/dataset_' + str(i) + '/processed/test/df.csv'


        df_training = pd.read_csv(_path_training)
        df_test = pd.read_csv(_path_test)
     

        _list_databases_training.append(df_training)
        _list_databases_test.append(df_test)
        _list_databases_name.append('database_' + str(i))



        i = i +1

    except:
        _error = True



# INPUT -- INPUT -- INPUT -- INPUT -- INPUT -- INPUT -- INPUT -- INPUT -- 
_list_simulation_sample_name = ['NSS', 'SPB','DEN', 'OUT']
for i_db in range(len(_list_databases_training)):

    _list_simulation_samples.append([])

    #NSS:
    _temp_list = sim.f_NSS(_list_databases_training[i_db])
    _list_simulation_samples[i_db].append(_temp_list)

    #SPB:
    _temp_list = sim.f_NSS(_list_databases_training[i_db])
    _list_simulation_samples[i_db].append(_temp_list)

    #DEN:
    _temp_list = sim.f_DEN(_list_databases_training[i_db])
    _list_simulation_samples[i_db].append(_temp_list)    

    #OUT:
    _temp_list = sim.f_OUT(_list_databases_training[i_db])
    _list_simulation_samples[i_db].append(_temp_list)    





# ML FRAMEWORK
for i_db in range(len(_list_databases_training)):     
    for i_simulation in range(len(_list_simulation_samples[i_db])):
        for i_model in range(len(_list_models)):            

            _db = _list_databases_training[i_db]
            _db_test = _list_databases_test[i_db]
            _samples = _list_simulation_samples[i_db][i_simulation]
            # _model = _list_models[i_model]

            _temp_X_columns = list(_db.loc[:,_db.columns.str.startswith("X")].columns)
    
            # print("db = ", i_db)
            # print("simulation = ", i_simulation)
            # print("model = ", i_model)
            # print("------------------\n")
            


#             #Framework de Accuracy
#             for i in range(len(_samples)):

#                 _samples_temp = _samples[0:i+1]

#                 #prepara X and y
#                 X_train = _db[_db['sample_id'].isin(_samples_temp)].loc[:,_temp_X_columns]
#                 y_train_true = _db[_db['sample_id'].isin(_samples_temp)].loc[:,'label']

#                 #FIT - train model:
#                 _list_models[i_model].fit(X_train, y_train_true)

#                 #db test (evaluation):
#                 X_test = _db_test.loc[:,_temp_X_columns]
#                 y_test_true = _db_test.loc[:,'label']

#                 #Predict:
#                 y_test_predict = _list_models[i_model].predict_proba()

#                 #Accuracy Results 
#                 _list_accuracy_on_labels_evaluated = accuracy_score(y_test_true, y_test_predict)
                    # LIS OT ACCURACY!!!!!!!



            # LITS -- outcome FROM FRAMEWORK OF ACCURACY!!!!!:
            _list_labels_evaluated = np.arange(1, 301, 1).tolist()
            _list_accuracy_on_labels_evaluated = random.sample(range(15), 15)
            
    
            #RESULTS OUTPUT:
            #name:
            _name_temp = _list_databases_name[i_db] + ' | ' + _list_simulation_sample_name[i_simulation] + ' | ' + _list_models_name[i_model]
            _results_output[0].append(_name_temp)            
            _results_output[1].append(_list_labels_evaluated)
            _results_output[2].append(_list_accuracy_on_labels_evaluated)                        



# Criar Pandas Dataset para depois gerar grafico!

# - Columns:
#     - Datanase (base 1, ... , base M
#     - Simulacao Number (SSP, NNS, ... etc)
#     - Model (Logistic, DecisionTre, ... )
#     - Order of Labels (1, 2... n)
#     - Labels Evaluated (5, 89, ..., random, , n)
#     - Accueracy of Label (1%, 2%, ... , 10%, ... 99%)