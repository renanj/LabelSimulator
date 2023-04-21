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
from faiss import StandardGpuResources, StandardGpuIndexFlatL2

import random
import math

from tqdm import tqdm


#WARNING: "index" and "sample_id" are completing different thngs... !

def closest_value(row):
    non_null_values = row.dropna()
    if non_null_values.empty:
        return None
    else:
        return non_null_values.iloc[0]



def f_run_simulations(df_embbedings, simulation_list = None):


    if simulation_list is None:
        # simulation_list = ['Random', 'Equal_Spread', 'Dense_Areas_First', 'Centroids_First', 'Cluster_Boarder_First',  'Outliers_First']
        simulation_list = ['Random', 'Equal_Spread']
    else:
        if 'Random' not in simulation_list:
            simulation_list.append('Random')
        else: 
            None


    #FAISS INDICES & DISTANCES DATAFRAME
    df_faiss_indices, df_faiss_distances = bblocks.f_faiss(df_embbedings)

    #BUILDING BLOCKS:
    _samples_id_list_random = bblocks.f_cold_start(df_embbedings)    
    _samples_id_list_ordered_SPB = bblocks.f_SPB(df_embbedings, df_faiss_distances, df_faiss_indices, _cold_start_samples_id=_samples_id_list_random)
    _samples_id_list_ordered_DEN = bblocks.f_den(df_embbedings, df_faiss_distances, df_faiss_indices, _cold_start_samples_id=_samples_id_list_random, k=5):
    _samples_id_list_ordered_OUT = bblocks.f_out(_samples_id_ordered_DEN)
    _samples_id_list_ordered_CLU = bblocks.f_clu(df_embbedings, k=None)

    

    #SIMULATION RUN based on "simulation_list":
    for _sim in simulation_list:

        if _sim == 'Random':
            print("Starting Random...")            
            _list_simulations_sample_id.append(_samples_id_list_random)
            _list_simulations_proceeded.append(_sim)
            print("End Random!")
            print("--------------------")


        elif _sim == 'Equal_Spread':
            print("Starting Equal Spread...")                        
            _samples_id_ordered = _samples_id_list_ordered_SPB
            _list_simulations_sample_id.append(_samples_id_ordered)
            _list_simulations_proceeded.append(_sim)
            print("End Equal Spread!")
            print("--------------------")


        elif _sim == 'Dense_Areas_First'
            print("Starting Dense_Areas_First...")            
            _samples_id_ordered = [val for pair in zip(_samples_id_list_ordered_SPB, _samples_id_list_ordered_DEN) for val in pair]
            _samples_id_ordered = list(set(_samples_id_ordered))
            _list_simulations_sample_id.append(_samples_id_ordered)
            _list_simulations_proceeded.append(_sim)
            print("End Dense_Areas_First!")
            print("--------------------")


        elif _sim == 'Centroids_First':
            print("Starting Centroids_First...")            
            _samples_id_ordered = [val for pair in zip(_samples_id_list_ordered_SPB, _samples_id_list_ordered_CLU) for val in pair]
            _samples_id_ordered = list(set(_samples_id_ordered))
            _list_simulations_sample_id.append(_samples_id_ordered)
            _list_simulations_proceeded.append(_sim)
            print("End Centroids_First!")
            print("--------------------")    


        # elif _sim == 'Cluster_Borders_First':
        #     print("Starting Cluster_Borders_First'...")            
        #     TBD... 

        #     _list_simulations_sample_id.append(_samples_id_ordered)
        #     _list_simulations_proceeded.append(_sim)
        #     print("End Cluster_Borders_First'!")
        #     print("--------------------")    


        elif _sim == 'Outliers_First'
            print("Starting Outliers_First...")            
            _samples_id_ordered = [val for pair in zip(_samples_id_list_ordered_SPB, _samples_id_list_ordered_OUT) for val in pair]
            _samples_id_ordered = list(set(_samples_id_ordered))
            _list_simulations_sample_id.append(_samples_id_ordered)
            _list_simulations_proceeded.append(_sim)
            print("End Outliers_First!")
            print("--------------------")

        
        else:
            print("We don't have a function ready for {} simulation!", _sim)
    return _list_simulations_proceeded, _list_simulations_sample_id            

