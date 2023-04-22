from re import S
import pandas as pd
import numpy as np
import os
import random
import math
import building_blocks as bblocks
from scipy.spatial import distance_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')



#WARNING: "index" and "sample_id" are completing different thngs... !


def f_run_simulations(df_embbedings, simulation_list = None):


    if simulation_list is None:
        # simulation_list = ['Random', 'Equal_Spread', 'Dense_Areas_First', 'Centroids_First', 'Cluster_Boarder_First',  'Outliers_First']
        simulation_list = ['Random', 'Equal_Spread', 'Dense_Areas_First', 'Centroids_First',  'Outliers_First']
        # simulation_list = ['Random', 'Equal_Spread']
    else:
        if 'Random' not in simulation_list:
            simulation_list.append('Random')
        else: 
            None


    #FAISS INDICES & DISTANCES DATAFRAME
    df_faiss_indices, df_faiss_distances = bblocks.f_faiss(df_embbedings)

    #BUILDING BLOCKS:
    _samples_id_list_random, _samples_id_list_random_cold_start = bblocks.f_cold_start(df_embbedings)    
    _samples_id_list_ordered_SPB = bblocks.f_SPB(df_embbedings, df_faiss_distances, df_faiss_indices, _cold_start_samples_id=_samples_id_list_random_cold_start)
    _samples_id_list_ordered_DEN = bblocks.f_den(df_embbedings, df_faiss_distances, df_faiss_indices, _cold_start_samples_id=_samples_id_list_random_cold_start, k=5)
    _samples_id_list_ordered_OUT = bblocks.f_out(_samples_id_list_ordered_DEN)
    _centroids_samples_id_list_ordered_CLU, _clusters_samples_id_list_of_lists_ordered_CLU = bblocks.f_clu(df_embbedings, num_clusters=None, num_iterations=25, gpu_index=True)


    
    _list_simulations_sample_id = []
    _list_simulations_proceeded = []

    #SIMULATION RUN based on "simulation_list":
    for _sim in simulation_list:

        if _sim == 'Random':
            print("Starting Random...")            
            _list_simulations_sample_id.append(_samples_id_list_random)
            _list_simulations_proceeded.append(_sim)
            print("Qtd Samples = ", len(_samples_id_list_random))
            print("End Random!")
            print("--------------------")


        elif _sim == 'Equal_Spread':
            print("Starting Equal Spread...")                        
            _samples_id_ordered = _samples_id_list_ordered_SPB
            _list_simulations_sample_id.append(_samples_id_ordered)
            _list_simulations_proceeded.append(_sim)
            print("Qtd Samples = ", len(_samples_id_ordered))
            print("End Equal Spread!")
            print("--------------------")


        elif _sim == 'Dense_Areas_First':
            print("Starting Dense_Areas_First...")            
            _samples_id_ordered = [val for pair in zip(_samples_id_list_ordered_SPB, _samples_id_list_ordered_DEN) for val in pair]
            _samples_id_ordered = list(set(_samples_id_ordered))
            _list_simulations_sample_id.append(_samples_id_ordered)
            _list_simulations_proceeded.append(_sim)
            print("Qtd Samples = ", len(_samples_id_ordered))
            print("End Dense_Areas_First!")
            print("--------------------")


        elif _sim == 'Centroids_First':
            print("Starting Centroids_First...")            
            _samples_id_ordered = [val for pair in zip(_samples_id_list_ordered_SPB, _centroids_samples_id_list_ordered_CLU) for val in pair]
            _samples_id_ordered = list(set(_samples_id_ordered))
            _list_simulations_sample_id.append(_samples_id_ordered)
            _list_simulations_proceeded.append(_sim)
            print("Qtd Samples = ", len(_samples_id_ordered))
            print("End Centroids_First!")
            print("--------------------")    


        # elif _sim == 'Cluster_Borders_First':
        #     print("Starting Cluster_Borders_First'...")            
            
        #     _samples_id_ordered_by_cluster = []
        #     for i in range(len(_clusters_samples_id_list_of_lists_ordered_CLU)):
        #         samples_id_in_cluster = _clusters_samples_id_list_of_lists_ordered_CLU[i]

        #         #SPB:
        #         df_temp = df_embbedings[df_embbedings['sample_id'].isin(samples_id_in_cluster)].copy()
        #         df_temp = df_temp.reset_index(drop=True)

        #         df_faiss_distances_temp = df_faiss_distances[df_faiss_distances.index.isin(samples_id_in_cluster)].copy()
        #         df_faiss_distances_temp = df_faiss_distances_temp.reset_index(drop=True)

        #         df_faiss_indices_temp = df_faiss_distances[df_faiss_distances.index.isin(samples_id_in_cluster)].copy()
        #         df_faiss_indices_temp = df_faiss_indices_temp.reset_index(drop=True)

        #         _temp_samples_id_list_random, _temp_samples_id_list_random_cold_start = bblocks.f_cold_start(df_temp)    
        #         _temp_samples_id_list_ordered_SPB = bblocks.f_SPB(df_temp, df_faiss_distances_temp, df_faiss_indices_temp, _cold_start_samples_id=_temp_samples_id_list_random_cold_start)

        #         #OUT:
        #         _temp_samples_id_list_ordered_DEN = bblocks.f_den(df_temp, df_faiss_distances_temp, df_faiss_indices_temp, _cold_start_samples_id=_temp_samples_id_list_random_cold_start, k=5)
        #         _temp_samples_id_list_ordered_OUT = bblocks.f_out(_temp_samples_id_list_ordered_DEN)                


        #         #SPB(50%) + OUT(50%) within Cluster:
        #         _temp_list = [val for pair in zip(_temp_samples_id_list_ordered_SPB, _temp_samples_id_list_ordered_OUT) for val in pair]
        #         _temp_list = list(set(_temp_list))                
        #         _samples_id_ordered_by_cluster.append(_temp_list)

            
            _samples_id_ordered = [val for pair in zip(*_samples_id_ordered_by_cluster) for val in pair]
            print("Len List = ", len(_samples_id_ordered))
            _samples_id_ordered = list(set(_samples_id_ordered))                
            print("Len List = ", len(_samples_id_ordered))

            _list_simulations_sample_id.append(_samples_id_ordered)
            _list_simulations_proceeded.append(_sim)
            print("Qtd Samples = ", len(_samples_id_ordered))
            print("End Cluster_Borders_First'!")
            print("--------------------")    


        elif _sim == 'Outliers_First':
            print("Starting Outliers_First...")            
            _samples_id_ordered = [val for pair in zip(_samples_id_list_ordered_SPB, _samples_id_list_ordered_OUT) for val in pair]
            _samples_id_ordered = list(set(_samples_id_ordered))
            _list_simulations_sample_id.append(_samples_id_ordered)
            _list_simulations_proceeded.append(_sim)
            print("Qtd Samples = ", len(_samples_id_ordered))
            print("End Outliers_First!")
            print("--------------------")

        
        else:
            print("We don't have a function ready for {} simulation!", _sim)
    return _list_simulations_proceeded, _list_simulations_sample_id            

