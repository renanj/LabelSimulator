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
from celluloid import Camera

import warnings
warnings.filterwarnings('ignore')

from scipy.spatial import distance_matrix

import building_blocks as bblocks

import os


#CUIDADO!: o return 'e o SAMPLE_ID e nao o index...


# def f_random(df):

#     #TBD

def f_NSS(df, sample_selector=None):

    # nearest_spatial_neighbors
    # Nearest Spatial Neighbors

    # It will give a list with order necessary to be choosed    
    # df: dataframe with columns 'sample_id', ['X1', 'X2', ... 'X'n], 'manual_label', 'label'
    # output: order of sample_ids to be selected

    _temp_X_columns = list(df.loc[:,df.columns.str.startswith("X")].columns)
    distances, indices = bblocks.func_NSN(_df=df, _columns=_temp_X_columns, _neighbors=df.shape[0] - 1)

    if sample_selector == None:
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



# def orchestrator_simulation(simulation_name):


#     if ... 




#     else "T"