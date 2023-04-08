import pandas as pd 
import numpy as np 
from cuml.manifold import TSNE
import os

import config as config
config = config.config

dim_reduction_list = ['t-SNE']


def f_dim_reduction(df, dim_r, n_dimensions=2):
  if dim_r == 't-SNE':
    #colunas X....
    _temp_X_columns = list(df.loc[:,df.columns.str.startswith("X")].columns)
    tsne = TSNE(n_components = n_dimensions)
    X_2dimensions = tsne.fit_transform(df.loc[:,_temp_X_columns])        
    X_2dimensions = X_2dimensions.rename(columns={0: 'X1', 1: 'X2'})
    # X_2dimensions[:,0], X_2dimensions[:,1]        
    df = pd.concat([df[['sample_id',	'name',	'labels',	'manual_label']], X_2dimensions], axis=1)
    	
    return df        
    
  else:
    print ("We don't have a dim_reduction algo with this name")
    
  



print('[INFO] Starting Dimension Reduction Script')
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
        #[FLAG] shoundl't have more than 2 file here...we are using MAX 1:2 folder<->files (train and validation)
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
                    print("run... ")
                    # print ('    ' + _files)
                    #print('.../' + _deep_learning_arq_sub_folder + '/' + _files)
                                        
            #Here Starts the Dim Reduction for Each DB
            #++++++ ++++++ ++++++ ++++++ ++++++ ++++++ ++++++ ++++++ ++++++ ++++++ ++++++                    
                    df = pd.read_pickle(db_paths[4] + '/' + _deep_learning_arq_sub_folders + '/' + _files) 
                    
                    for dim_r in dim_reduction_list:                        
                        df = f_dim_reduction(df, dim_r)  
                        #Check if Folder Exist
                        if not os.path.exists(db_paths[4] +'/' + _deep_learning_arq_sub_folders + '__' +  dim_r):
                          os.makedirs(db_paths[4] +'/' + _deep_learning_arq_sub_folders + '__' +  dim_r)
                        df.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folders + '__' +  dim_r + '/' + 'df_' + config._list_train_val[i_train_val] + '.pkl')                                                            
                    print('---------------/n/n')