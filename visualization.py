from cuml.manifold import TSNE

# change feature extraxtion folder to vgg_16__{{RAW}} WHEN THERE Is no dim reduction


# https://medium.com/rapids-ai/tsne-with-gpus-hours-to-seconds-9d9c17c941db



# 1) Read .pkl on feature_extraction folder
# 2) run tsn-e and:
    # a) export a .pkl in the same db with structure 'arq_name_{{visualization_mode}}' - e.g.: "vgg_16_tsn_e.pkl"
    # b) 

dim_reduction_list = ['t-SNE']

#Plancton, mnist, etc...
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
                        df.to_pickle(db_paths[4] +'/' + _deep_learning_arq_sub_folders + '__' +  dim_r + '/' + 'df_' + config._list_train_val[i_train_val] + '.pkl')                                                            
                    print('---------------/n/n')                    



def f_dim_reduction(df, dim_r):
    if dim_r == 't-SNE':
        #colunas X....
        _temp_X_columns = list(df.loc[:,df.columns.str.startswith("X")].columns)
        tsne = TSNE(n_components = 2)
        X_2dimensions = tsne.fit_transform(df.loc[:,_temp_X_columns])        
        # X_2dimensions[:,0], X_2dimensions[:,1]        
        df = pd.concat([df, pd.DataFrame(arr, columns=["X1", "X2"])], axis=1
        return df 
    else:
        print ("We don't have a dim_reduction algo with this name")
        return None
