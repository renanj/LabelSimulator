from re import S
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#K-means
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs
from sklearn import datasets

from sklearn import neighbors, datasets
from sklearn.manifold import TSNE

import random
# from celluloid import Camera

import warnings
warnings.filterwarnings('ignore')

from scipy.spatial import distance_matrix

import building_blocks as bblocks

import os
import faiss
import random
import math


#WARNING: "index" and "sample_id" are completing different thngs... !

def closest_value(row):
    non_null_values = row.dropna()
    if non_null_values.empty:
        return None
    else:
        return non_null_values.iloc[0]

def f_run_simulations(df_embbedings, simulation_list = None):

    
    if simulation_list == None:
        simulation_list = ['Random', 'Equal_Spread', 'Dense_Areas_First', 'Centroids_First', 'Cluster_Boarder_First',  'Outliers_First']
    else:
        if 'Random' not in simulation_list:
            simulation_list.append('Random')
        else: 
            None


    #generate array with Embbedings info ("X1, X2, X3..." columns)
    _temp_X_columns = list(df_embbedings.loc[:,df_embbedings.columns.str.startswith("X")].columns)
    if _temp_X_columns == None:
      _temp_X_columns = list(df_embbedings.loc[:,df_embbedings.columns.str.startswith("X")].columns)

    X = np.ascontiguousarray(df_embbedings[_temp_X_columns].values.astype('float32'))
    d = X.shape[1]
    
    #calculate FAISS index... 
    index = faiss.IndexFlatL2(d)
    #Settings the IDs based on sample_id column & creating the correct "index"
    sample_ids = df_embbedings['sample_id'].values
    index = faiss.IndexIDMap(index)    
    # index.add(X)    
    index.add_with_ids(X, sample_ids) #creating the index with the sample_ids instead of sequential value
    #For reference: https://github.com/facebookresearch/faiss/wiki/Pre--and-post-processing
    _neighbors = df_embbedings.shape[0]
    faiss_distances, faiss_indices = index.search(X, _neighbors)

    #Generate DFs for Faiss Indices & Distance
    df_faiss_indices = pd.DataFrame(faiss_indices, index=sample_ids, columns=sample_ids)
    df_faiss_distances = pd.DataFrame(faiss_distances, index=sample_ids, columns=sample_ids)

    
    #Specific for Random -- we will used for Random as simulation, but also for cold start
    _random_samples_index = random.sample(range(df_embbedings.shape[0]),df_embbedings.shape[0])
    _random_samples_id = list(df_embbedings['sample_id'].iloc[_random_samples_index])

    

    _list_simulations_proceeded = []
    _list_simulations_sample_id = []


    #Run each Simulation on "simulation_list":
    for _sim in simulation_list:


        if _sim == 'Random':

            
            #Output
            # -------------------------
            _list_simulations_sample_id.append(_random_samples_id)                    
            _list_simulations_proceeded.append(_sim)


        elif _sim == 'Equal_SpreadE':
            
            #Cold Start:
            #at least 20% or min 50 samples will need to be labeded on cold start
            if len(_random_samples_id) >= 500:
                _cold_start_samples_id  = _random_samples_id[0:50]
            else:
                _cold_start_samples_id  = _random_samples_id[0:math.ceil(0.2*len(_random_samples_id))] # 20% of dataset


            ###### BEGINNING OF FUNCTION ###### ###### ###### ###### ###### ###### ###### ###### ######

            # Set the random seed for reproducibility

            # Initialize Vc and Vt as indices
            label_samples_id = _cold_start_samples_id
            unlabel_samples_id = list(df_embbedings['sample_id'][~df_embbedings['sample_id'].isin(_cold_start_samples_id)])


            # Initialize the list of selected sample indices during the script
            selected_sample_id = []            


            while len(unlabel_samples_id) >0:
               

                # define excluded elements
                excluded_elements = label_samples_id
                # create a mask where True corresponds to cells containing the excluded elements
                mask = df_faiss_indices.isin(excluded_elements)
                # create a new DataFrame with the values excluding excluded elements
                df_faiss_indices_with_NaN = df_faiss_indices.mask(mask)
                # check if length of df and result is the same
                if len(df_faiss_indices) != len(df_faiss_indices_with_NaN):
                    df_faiss_indices_with_NaN = df_faiss_indices_with_NaN.reindex(df_faiss_indices.index)
                #Replace first column with NaN
                df_faiss_indices_with_NaN.iloc[:,0] = np.NaN

                df_mask_boolean = df_faiss_indices_with_NaN.iloc[:,:].fillna(0)
                df_mask_boolean = df_mask_boolean.applymap(lambda x: 0 if x == 0 else 1)
                # df_mask_true_false = df_mask_boolean.applymap(lambda x: False if x == 0 else True)

                df_faiss_distances_with_NaN = df_faiss_distances * df_mask_boolean
                df_faiss_distances_with_NaN = df_faiss_distances_with_NaN.replace(0, np.NaN)        

                df_faiss_indices_with_NaN = df_faiss_indices_with_NaN[df_faiss_indices_with_NaN.index.isin(label_samples_id)]
                df_faiss_distances_with_NaN = df_faiss_distances_with_NaN[df_faiss_distances_with_NaN.index.isin(label_samples_id)]


                df_faiss_indices_with_NaN['closest_value'] = df_faiss_indices_with_NaN.apply(closest_value, axis=1)
                df_faiss_distances_with_NaN['closest_value'] = df_faiss_distances_with_NaN.apply(closest_value, axis=1)

                df_faiss_indices_with_NaN_result = df_faiss_indices_with_NaN.loc[:,'closest_value']
                # df_faiss_distances_with_NaN_result = df_faiss_distances_with_NaN.loc[:,'closest_value']
                # df_faiss_indices_with_NaN_result[df_faiss_distances_with_NaN_result.idxmax()]

                sample_selected_result = df_faiss_indices_with_NaN_result[df_faiss_distances_with_NaN.loc[:,'closest_value'].idxmax()]
                sample_selected_result = int(sample_selected_result)
                

                #Add the sample_id in label_sample_ids and remove from unlabeled_sample_ids                
                unlabel_samples_id = np.delete(unlabel_samples_id, np.where(unlabel_samples_id == sample_selected_result))
                label_samples_id = np.concatenate([label_samples_id , np.array([sample_selected_result])])                
                                             

                #Add to the selected_sample_id
                selected_sample_id.append(sample_selected_result)

                
            ###### END OF FUNCTION ###### ###### ###### ###### ###### ###### ###### ###### ######

            #Output
            # -------------------------
            temp_list = _cold_start_samples_id + selected_sample_id

            _list_simulations_sample_id.append(temp_list)                                
            _list_simulations_proceeded.append(_sim)


        elif _sim == 'old_version_Equal_Spread':
            
            #Cold Start:
            #at least 20% or min 50 samples will need to be labeded on cold start
            if len(_random_samples_id) >= 500:
                _cold_start_samples_id  = _random_samples_id[0:50]
            else:
                _cold_start_samples_id  = _random_samples_id[0:math.ceil(0.2*len(_random_samples_id))] # 20% of dataset


            ###### BEGINNING OF FUNCTION ###### ###### ###### ###### ###### ###### ###### ###### ######

            # Set the random seed for reproducibility

            # Initialize Vc and Vt as indices
            label_samples_id = _cold_start_samples_id
            unlabel_samples_id = list(df_embbedings['sample_id'][~df_embbedings['sample_id'].isin(_cold_start_samples_id)])


            # Initialize the list of selected sample indices during the script
            selected_sample_id = []            


            while len(unlabel_samples_id) >0:


                df_embbedings_label = df_embbedings[df_embbedings['sample_id'].isin(label_samples_id)].copy()
                df_embbedings_label = df_embbedings_label.reset_index(drop=True)
                X_label_samples = np.ascontiguousarray(df_embbedings_label[_temp_X_columns].values.astype('float32'))
                d_label_samples = X_label_samples.shape[1]


                df_embbedings_unlabel = df_embbedings[~df_embbedings['sample_id'].isin(label_samples_id)].copy()
                df_embbedings_unlabel = df_embbedings_unlabel.reset_index(drop=True)
                X_unlabel_samples = np.ascontiguousarray(df_embbedings_unlabel[_temp_X_columns].values.astype('float32'))
                d_unlabel_samples = X_unlabel_samples.shape[1]                



                #calculate FAISS index... 
                index_label_samples = faiss.IndexFlatL2(d_label_samples)
                index_label_samples.add(X_label_samples)    

                # _neighbors_labels = df_embbedings_label.shape[0]
                _query_unlabel = np.ascontiguousarray(df_embbedings_unlabel[_temp_X_columns].values.astype('float32'))


                #Min distance
                min_distances, min_indices = index_label_samples.search(_query_unlabel, k=1)
                
                # Find the index of the sample in min_distance array with the maximum minimum distance to labeled
                index_unlabel_max_min_dist = np.argmax(min_distances)
                # Using the index found previously, find the correspondet sample_id in unlabeled_df                            
                result_max_min_sample_id = df_embbedings_unlabel['sample_id'][df_embbedings_unlabel.index ==index_unlabel_max_min_dist]
                

                #Add the sample_id in label_sample_ids and remove from unlabeled_sample_ids                
                unlabel_samples_id = np.delete(unlabel_samples_id, result_max_min_sample_id.index[0])                
                label_samples_id = np.concatenate((label_samples_id , [result_max_min_sample_id.values[0]]))                                

                #Add to the selected_sample_id
                selected_sample_id.append(result_max_min_sample_id.values[0])

                
            ###### END OF FUNCTION ###### ###### ###### ###### ###### ###### ###### ###### ######

            #Output
            # -------------------------
            temp_list = _cold_start_samples_id + selected_sample_id

            _list_simulations_sample_id.append(temp_list)                                
            _list_simulations_proceeded.append(_sim)






        elif _sim == 'Dense_Areas_First' or _sim == 'Outliers_First':
            

            #[TO-DO] Falta incluir o SPB e balancear com o DEN (50% and 50%)

            k = 5            
            den_scores_array = -np.sum(faiss_distances[:, :k]**2, axis=-1) / k
            den_scores_list = list(den_scores_array)
            den_index_list = range(len(den_scores_list))
            #create temp_df, sort by score descending order and use den_index to capture the sample_id
            

            temp_df = df_embbedings[['sample_id']].copy()
            temp_df['den_score'] = den_scores_array



            if _sim == 'Dense_Areas_First':
                temp_df = temp_df.sort_values(by='den_score', ascending=False).reset_index(drop=True)
            else:
                temp_df = temp_df.sort_values(by='den_score', ascending=True).reset_index(drop=True)


            #Output
            # -------------------------
            _list_simulations_sample_id.append(list(temp_df['sample_id']))
            _list_simulations_proceeded.append(_sim)




        elif _sim == 'Centroids_First':
            #Run DEN





            #Output
            # -------------------------
            _list_simulations_sample_id.append(    )                    
            _list_simulations_proceeded.append(_sim)



        elif _sim == 'Cluster_Boarder_First':
            #Run OUT



            #Output
            # -------------------------
            _list_simulations_sample_id.append(    )                    
            _list_simulations_proceeded.append(_sim)



        # elif _sim == 'Outliers_First':
        #     #Run OUT            



        #     #Output
        #     # -------------------------
        #     _list_simulations_sample_id.append(    )                    
        #     _list_simulations_proceeded.append(_sim)            

        else:
            print("We don't have a function ready for {} simulation!", _sim)




def f_NSS(df, sample_selector=None):

    _temp_X_columns = list(df.loc[:,df.columns.str.startswith("X")].columns)
    distances, indices = bblocks.func_NSN(_df=df, _columns=_temp_X_columns, _neighbors=df.shape[0] - 1)

    sample_selector = random.randrange(*sorted([0,df.shape[0]]))
    
    indices_from_sample = np.insert(indices[sample_selector], 0, sample_selector)

    return list(indices_from_sample)



def f_SPB(df, samples_with_label=None):

    # spatial_balancing        
    # Spatial Balancing        

    # It will give a list with order necessary to be choosed    
    # df: dataframe with columns 'sample_id', ['X1', 'X2', ... 'X'n], 'manual_label', 'label'
    # output: order of sample_ids to be selected

    if samples_with_label == None:
        samples_with_label = []
        samples_with_label.append(random.randrange(*sorted([1,df.shape[0]+2])))

    Vc = df[~df['sample_id'].isin(samples_with_label)].reset_index(drop=True)
    Vt = df[df['sample_id'].isin(samples_with_label)].reset_index(drop=True)
    Vt['manual_label'] = Vt['labels']

    _temp_X_columns = list(df.loc[:,df.columns.str.startswith("X")].columns)

    # _sample_id = random.randrange(0,Vt.shape[0]-1)
    # df['label'][df['sample_id'] == _sample_id] = 2


    # Calcular matrix de distancia
    data = df.loc[:,_temp_X_columns].values.tolist()
    df_pre_matrix = pd.DataFrame(data, columns=_temp_X_columns, index=df['sample_id'].values)
    df_matrix = pd.DataFrame(distance_matrix(df_pre_matrix.values, df_pre_matrix.values), index=df_pre_matrix.index, columns=list(df_pre_matrix.index.values))

    #Vc = index
    #Vt = columns


    _list_choices = samples_with_label.copy()    
    for i in range(Vc.shape[0] - 1):        

        Vc = df[~df['sample_id'].isin(samples_with_label)].reset_index(drop=True)
        Vt = df[df['sample_id'].isin(samples_with_label)].reset_index(drop=True)

        df_matrix_temp_del = df_matrix[df_matrix.index.isin(Vc['sample_id'].values)].loc[:,Vt['sample_id'].values]        
        _can = df_matrix_temp_del.min(axis=1).idxmax(axis=0)


        # _max_dis, _can = bblocks.func_SPB(Vc.loc[:,_temp_X_columns].values, Vt.loc[:,_temp_X_columns].values, sample_id_list=Vc['sample_id'].values)     

        samples_with_label.append(_can)
        _list_choices.append(_can)   

        print(len(_list_choices))


    _list_choices.append(df['sample_id'][~df['sample_id'].isin(_list_choices)].values[0])
    return _list_choices


    # _list_choices = []
    # for i in range(Vc.shape[0] - 1):   

    #     Vc = df[~df['sample_id'].isin(samples_with_label)].reset_index(drop=True)
    #     Vt = df[df['sample_id'].isin(samples_with_label)].reset_index(drop=True)

    #     _max_dis, _can = bblocks.func_SPB(Vc.loc[:,_temp_X_columns].values, Vt.loc[:,_temp_X_columns].values, sample_id_list=Vc['sample_id'].values)     

    #     samples_with_label.append(_can)
    #     _list_choices.append(_can)        
        

    # return _list_choices

def f_DEN(df, _k =5):

    # density_estimation
    # Density Estimation

    # output: order of sample_ids to be selected


    _temp_X_columns = list(df.loc[:,df.columns.str.startswith("X")].columns)

    _max_DEN = None
    _max_DEN_index = None
    _list_sample_id = []
    _list_DEN = []

    for i in range(df.shape[0]):
        
        temp_id = df.loc[i,'sample_id']
        temp_DEN = bblocks.func_DEN(df=df, columns=_temp_X_columns, sample_id=temp_id, k=_k)        
        _list_sample_id.append(temp_id) 
        _list_DEN.append(temp_DEN)
        
        if _max_DEN == None:
            _max_DEN = temp_DEN
            _max_DEN_index = temp_id
            
        else:
            if _max_DEN < temp_DEN:
                _max_DEN = temp_DEN
                _max_DEN_index = temp_id


        _temp_df = pd.DataFrame(list(zip(_list_sample_id, _list_DEN)),
                    columns =['sample_id_order', 'DEN_value'])
        _temp_df = _temp_df.sort_values(by='DEN_value', ascending=True)
        _temp_df = _temp_df.reset_index(drop=True)
        
    return list(_temp_df['sample_id_order'])



def f_OUT(df, _k=5):

    # outlier_detection
    # Outlier Detection

    # output: order of sample_ids to be selected

    _temp_X_columns = list(df.loc[:,df.columns.str.startswith("X")].columns)

    _max_DEN = None
    _max_DEN_index = None
    _list_sample_id = []
    _list_DEN = []


    for i in range(df.shape[0]):
        
        temp_id = df.loc[i,'sample_id']
        temp_DEN = bblocks.func_OUT(df=df, columns=_temp_X_columns, sample_id=temp_id, threshold=10, k=5)
        _list_sample_id.append(temp_id) 
        _list_DEN.append(temp_DEN)    
        
        if _max_DEN == None:
            _max_DEN = temp_DEN
            _max_DEN_index = temp_id
            
        else:
            if _max_DEN < temp_DEN:
                _max_DEN = temp_DEN
                _max_DEN_index = temp_id



        _temp_df = pd.DataFrame(list(zip(_list_sample_id, _list_DEN)),
                    columns =['sample_id_order', 'DEN_value'])
        _temp_df = _temp_df.sort_values(by='DEN_value', ascending=True)
        _temp_df = _temp_df.reset_index(drop=True)
        
    return list(_temp_df['sample_id_order'])
