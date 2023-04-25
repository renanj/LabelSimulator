import pandas as pd
import numpy as np
import faiss
from faiss import StandardGpuResources
from tqdm import tqdm


import config as config
config = config.config




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
        res = faiss.StandardGpuResources() #index = faiss.IndexFlatL2(d)
        index = faiss.GpuIndexFlatL2(res, X.shape[1])
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
                    print ("run Faiss for...     ", _files)                    
                    
                    
            #Here Starts the Simulation for Each DB        
            #++++++ ++++++ ++++++ ++++++ ++++++ ++++++ ++++++ ++++++ ++++++ ++++++ ++++++                    
                     
                    df = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folders + '/' + _files)
					df_faiss_indices, df_faiss_distances = f_faiss(df, _GPU_flag=True):

					df_faiss_indices.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folders + '/' + 'df_faiss_indices_' + config._list_train_val[i_train_val]  + '.pkl')
					df_faiss_distances.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folders + '/' + 'df_faiss_distances_' + config._list_train_val[i_train_val]  + '.pkl')