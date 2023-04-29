import pandas as pd
import numpy as np
import os

from faiss import StandardGpuResources
import faiss
from tqdm import tqdm
import simulation as sim
import config as config
config = config.config
import sys
import cudf



run_simulation_option = sys.argv[1]

def f_faiss(df_embbedings, _GPU_flag=True):

    print("[FAISS] - Start")

    # 1) generate array with Embbedings info ("X1, X2, X3..." columns)
    _temp_X_columns = list(df_embbedings.loc[:,df_embbedings.columns.str.startswith("X")].columns)
    if _temp_X_columns == None:
      _temp_X_columns = list(df_embbedings.loc[:,df_embbedings.columns.str.startswith("X")].columns)

    X = np.ascontiguousarray(df_embbedings[_temp_X_columns].values.astype('float32'))
    d = X.shape[1]
        
    # 2) calculate FAISS index...  
    if _GPU_flag is True:   
        print("[Using GPU...")
        # res = faiss.StandardGpuResources() 
        # index = faiss.GpuIndexFlatL2(res, X.shape[1])
        index = faiss.IndexFlatL2(d)

    else:
        index = faiss.IndexFlatL2(d)


    # 3) Settings the IDs based on sample_id column & creating the correct "index"
    sample_ids = df_embbedings['sample_id'].values
    index = faiss.IndexIDMap(index)        
    index.add_with_ids(X, sample_ids) # index.add(X) #creating the index with the sample_ids instead of sequential value #For reference: https://github.com/facebookresearch/faiss/wiki/Pre--and-post-processing    
    _neighbors = df_embbedings.shape[0]
    faiss_distances, faiss_indices = index.search(X, _neighbors)


    # 4) Generate DFs for Faiss Indices & Distance
    df_faiss_indices = pd.DataFrame(faiss_indices, index=sample_ids, columns=sample_ids)
    df_faiss_distances = pd.DataFrame(faiss_distances, index=sample_ids, columns=sample_ids)

    print("[FAISS] - End")
    return df_faiss_indices, df_faiss_distances





#Plancton, mnist, etc...
print('[INFO] Starting Faiss')
for db_paths in config._list_data_sets_path:

    print("\n\nPATH -------")
    print('=====================')
    print(db_paths[0])
    print('=====================')
    print('=====================\n')    
    _deep_learning_arq_sub_folder =  [db_paths for db_paths in os.listdir(db_paths[4]) if not db_paths.startswith('.')]    
    for _deep_learning_arq_sub_folder in _deep_learning_arq_sub_folder:
        print('-------')
        print('.../' + _deep_learning_arq_sub_folder)
        #list of files
        _list_files = [_files for _files in os.listdir(db_paths[4] + '/' + _deep_learning_arq_sub_folder) if not _files.startswith('.')]
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
                    print ("run Faiss for...     ", _files)                                                                        
                     
                    df = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folder + '/' + _files)
                    
                    df_faiss_indices, df_faiss_distances = f_faiss(df, _GPU_flag=True)

                    df_faiss_indices.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder + '/' + 'df_faiss_indices_' + config._list_train_val[i_train_val]  + '.pkl')
                    df_faiss_distances.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder + '/' + 'df_faiss_distances_' + config._list_train_val[i_train_val]  + '.pkl')


                    
                    
                    if run_simulation_option == 'yes':
                        print("[INFO] will run Simulation too")
                        print("should be on GPU...")
                        df = cudf.DataFrame.from_pandas(df)
                        df_faiss_indices = cudf.DataFrame.from_pandas(df_faiss_indices)
                        df_faiss_distances = cudf.DataFrame.from_pandas(df_faiss_distances)


                        _list_simulation_sample_name, _list_simulation_ordered_samples_id = sim.f_run_simulations(df_embbedings = df, df_faiss_indices=df_faiss_indices, df_faiss_distances=df_faiss_distances, simulation_list = None)                        

                        _simulation_order_df = pd.DataFrame(_list_simulation_ordered_samples_id ,columns=_list_simulation_sample_name)                                                        
                        _simulation_order_df.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folder + '/' + 'df_simulation_ordered_' + config._list_train_val[i_train_val]  + '.pkl')
                    
