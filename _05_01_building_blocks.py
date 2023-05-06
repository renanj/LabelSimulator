import pandas as pd
import numpy as np
import random
import math
import warnings
warnings.filterwarnings('ignore')
import faiss
from faiss import StandardGpuResources
from tqdm import tqdm
import cupy as cp
import cudf

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def closest_value(row):
    non_null_values = row.dropna()
    if non_null_values.empty:
        return None
    else:
        return non_null_values.iloc[0]


def f_cold_start(df_embbedings, _random_state=42):

    random.seed(_random_state)
    _random_samples_index = random.sample(range(df_embbedings.shape[0]),df_embbedings.shape[0])
    _random_samples_id = list(df_embbedings['sample_id'].iloc[_random_samples_index])

    if len(_random_samples_id) >= 500:
        _cold_start_samples_id  = _random_samples_id[0:50]
    else:
        _cold_start_samples_id  = _random_samples_id[0:math.ceil(0.2*len(_random_samples_id))]

    return _random_samples_id, _cold_start_samples_id



def f_SPB(df_embbedings, df_faiss_distances, df_faiss_indices, _cold_start_samples_id=None):


  if _cold_start_samples_id is None: 
      _cold_start_samples_id = f_cold_start(df_embbedings)

  array_labels_sample_ids = cp.array(_cold_start_samples_id)
  array_unlabels_sample_ids = cp.array(df_embbedings['sample_id'][~df_embbedings['sample_id'].isin(_cold_start_samples_id)])            

  array_faiss_indices = cp.array(df_faiss_indices)
  array_faiss_distances = cp.array(df_faiss_distances)

  while len(array_unlabels_sample_ids) > 0:            
    array_mask_false_true = cp.isin(array_faiss_indices, array_labels_sample_ids)
    array_faiss_indices_filtered = cp.where(~array_mask_false_true, array_faiss_indices, cp.nan)      
    indices_to_be_filtered = cp.argpartition(cp.isnan(array_faiss_indices_filtered), 1, axis=1)[:, 0]
    result_indices = array_faiss_indices[cp.arange(len(array_faiss_indices)), indices_to_be_filtered]
    result_distance = array_faiss_distances[cp.arange(len(array_faiss_distances)), indices_to_be_filtered]
    selected_sample_id = result_indices[cp.argmax(result_distance)]    

    array_unlabels_sample_ids = array_unlabels_sample_ids[array_unlabels_sample_ids != selected_sample_id]
    array_labels_sample_ids= cp.append(array_labels_sample_ids, selected_sample_id)
    # selected_sample_mask = (array_unlabels_sample_ids == selected_sample_id)    
    # array_unlabels_sample_ids = array_unlabels_sample_ids[~selected_sample_mask]
    # array_labels_sample_ids = cp.append(array_labels_sample_ids, array_unlabels_sample_ids[selected_sample_mask])          

  ordered_selected_samples_id = array_labels_sample_ids.tolist()      

  return ordered_selected_samples_id

    

def f_den(df_embbedings, df_faiss_distances, df_faiss_indices, _cold_start_samples_id=None, k=6):

    _array_with_values = df_faiss_distances.iloc[:,1:k].values
    den_scores_array = -np.sum(_array_with_values**2, axis=-1) / k 

    
    temp_df = df_embbedings[['sample_id']].copy()
    temp_df['den_score'] = den_scores_array

    temp_df = temp_df.sort_values(by='den_score', ascending=False).reset_index(drop=True)


    ordered_selected_samples_id = list(temp_df['sample_id'].values)
    return ordered_selected_samples_id


def f_out(desentity_ordered_selected_samples_id):

    print("Before:")
    print("First 5 = ", desentity_ordered_selected_samples_id[:5])
    print("Final 5 = ", desentity_ordered_selected_samples_id[:-5])     
    desentity_ordered_selected_samples_id_new = desentity_ordered_selected_samples_id.reverse().copy()
    print("After:")
    print("First 5 = ", desentity_ordered_selected_samples_id_new[:5])
    print("Final 5 = ", desentity_ordered_selected_samples_id_new[:-5])    
    return desentity_ordered_selected_samples_id_new



def f_clu(df_embbedings, num_clusters=None, num_iterations=15, gpu_index=True):

    if num_clusters is None:
        num_clusters = round(df_embbedings.shape[0] * 0.15)

    _temp_X_columns = list(df_embbedings.loc[:,df_embbedings.columns.str.startswith("X")].columns)
    if _temp_X_columns == None:
      _temp_X_columns = list(df_embbedings.loc[:,df_embbedings.columns.str.startswith("X")].columns)

    X = np.ascontiguousarray(df_embbedings[_temp_X_columns].values.astype('float32'))
    d = X.shape[1]


    res = faiss.StandardGpuResources() #index = faiss.IndexFlatL2(d)
    index = faiss.GpuIndexFlatL2(res, X.shape[1])


    sample_ids = df_embbedings['sample_id'].values
    index = faiss.IndexIDMap(index)        
    index.add_with_ids(X, sample_ids)


    # Run k-means clustering
    kmeans = faiss.Kmeans(X.shape[1], num_clusters, niter=num_iterations, verbose=True)
    kmeans.train(X)
    centroids = kmeans.centroids

    # Assign one (closest) samples for each Centroid
    D, I = index.search(centroids, 1)
    centroids_sample_ids = list(I.reshape(1, -1)[0])
    df_embbedings['centroid'] = False 
    df_embbedings['centroid'][df_embbedings['sample_id'].isin(centroids_sample_ids)] = True


    # Assign cluster labels to each sample
    _, labels = kmeans.index.search(X, 1)
    # Merge the labels with the original dataframe
    df_embbedings["kmeans_label"] = labels


    # Store the sample_ids for each kmeans_label in a dictionary
    label_dict = {}
    for i, label in enumerate(labels):
        if label[0] not in label_dict:
            label_dict[label[0]] = []
        label_dict[label[0]].append(sample_ids[i])

    # Convert the dictionary to a list of lists
    sample_ids_list = [label_dict[label] for label in range(num_clusters)]    

    # return centroids_sample_ids, label_dict, sample_ids_list, df_embbedings
    return centroids_sample_ids, sample_ids_list