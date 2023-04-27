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

try:
    import cudf
except ImportError:
    print("Not possible to import cudf")





from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def generate_data(n_samples, cluster_std_ratio, n_outliers):
    # Generate data with 2 clusters where cluster 1 is more disperse than cluster 2
    X, y = make_blobs(n_samples=n_samples-n_outliers, centers=2, cluster_std=[1.5*cluster_std_ratio, 0.5], random_state=42)
    outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, 2))
    X = np.concatenate((X, outliers), axis=0)
    y = np.concatenate((y, np.full((n_outliers,), fill_value=-1)), axis=0)

    # Create a dataframe with X1 and X2
    df = pd.DataFrame(X, columns=['X1', 'X2'])
    df['labels'] = y
    df['manual_label'] = "-"
    df['sample_id'] = range(1, n_samples+1)  # Add sample IDs starting from 1
    return df


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

  while len(array_labels_sample_ids) < 100:            
      array_mask_false_true = cp.isin(array_faiss_indices, array_labels_sample_ids)
      array_faiss_indices_filtered = cp.where(~array_mask_false_true, array_faiss_indices, cp.nan)      
      indices_to_be_filtered = cp.argpartition(cp.isnan(array_faiss_indices_filtered), 1, axis=1)[:, 0]
      result_indices = array_faiss_indices[cp.arange(len(array_faiss_indices)), indices_to_be_filtered]
      result_distance = array_faiss_distances[cp.arange(len(array_faiss_distances)), indices_to_be_filtered]
      selected_sample_id = result_indices[cp.argmax(result_distance)]
      array_unlabels_sample_ids = array_labels_sample_ids[array_labels_sample_ids != selected_sample_id]
      array_labels_sample_ids= cp.append(array_unlabels_sample_ids, selected_sample_id)

  return list(array_labels_sample_ids) 

    

def f_den(df_embbedings, df_faiss_distances, df_faiss_indices, _cold_start_samples_id=None, k=5):

    _array_with_values = df_faiss_distances.iloc[:,:k].values
    den_scores_array = -np.sum(_array_with_values**2, axis=-1) / k 

    
    temp_df = df_embbedings[['sample_id']].copy()
    temp_df['den_score'] = den_scores_array

    temp_df = temp_df.sort_values(by='den_score', ascending=False).reset_index(drop=True)



    ordered_selected_samples_id = list(temp_df['sample_id'].values)
    return ordered_selected_samples_id


def f_out(desentity_ordered_selected_samples_id):

    # print("Before:")
    # print("First 5 = ", desentity_ordered_selected_samples_id[:5])
    # print("Final 5 = ", desentity_ordered_selected_samples_id[:-5])     
    desentity_ordered_selected_samples_id.reverse()
    # print("After:")
    # print("First 5 = ", desentity_ordered_selected_samples_id[:5])
    # print("Final 5 = ", desentity_ordered_selected_samples_id[:-5])    
    return desentity_ordered_selected_samples_id



def f_clu(df_embbedings, num_clusters=None, num_iterations=25, gpu_index=True):

    if num_clusters is None:
        num_clusters = round(df_embbedings.shape[0] * 0.20)

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