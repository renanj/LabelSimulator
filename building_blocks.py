import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
import faiss
from faiss import StandardGpuResources, StandardGpuIndexFlatL2

def closest_value(row):
    non_null_values = row.dropna()
    if non_null_values.empty:
        return None
    else:
        return non_null_values.iloc[0]


def f_faiss(df_embbedings):

    print("[FAISS] - Start")

    # 1) generate array with Embbedings info ("X1, X2, X3..." columns)
    _temp_X_columns = list(df_embbedings.loc[:,df_embbedings.columns.str.startswith("X")].columns)
    if _temp_X_columns == None:
      _temp_X_columns = list(df_embbedings.loc[:,df_embbedings.columns.str.startswith("X")].columns)

    X = np.ascontiguousarray(df_embbedings[_temp_X_columns].values.astype('float32'))
    d = X.shape[1]

    
    # 2) calculate FAISS index...     
    res = faiss.StandardGpuResources() #index = faiss.IndexFlatL2(d)
    index = faiss.GpuIndexFlatL2(res, X.shape[1])

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



def f_cold_start(df_embbedings, _random_state=42):

    random.seed(_random_state)
    _random_samples_index = random.sample(range(df_embbedings.shape[0]),df_embbedings.shape[0])
    _random_samples_id = list(df_embbedings['sample_id'].iloc[_random_samples_index])

    if len(_random_samples_id) >= 500:
        _cold_start_samples_id  = _random_samples_id[0:50]
    else:
        _cold_start_samples_id  = _random_samples_id[0:math.ceil(0.2*len(_random_samples_id))]

    return _cold_start_samples_id




def f_SPB(df_embbedings, df_faiss_distances, df_faiss_indices, _cold_start_samples_id=None):

    if _cold_start_samples_id == None: 
        _cold_start_samples_id = f_cold_start(df_embbedings)

    #Initiatize Labels and Unlabels Samples
    label_samples_id = np.array(_cold_start_samples_id)
    unlabel_samples_id = np.array(df_embbedings['sample_id'][~df_embbedings['sample_id'].isin(_cold_start_samples_id)])            

    # Initialize the list of selected sample indices during the script
    selected_sample_id = []            


    with tqdm(total=len(unlabel_samples_id)) as pbar:            
        while len(unlabel_samples_id) >0:
           

            
            excluded_elements = label_samples_id # define excluded elements            
            mask = df_faiss_indices.isin(excluded_elements) # create a mask where True corresponds to cells containing the excluded elements            
            df_faiss_indices_with_NaN = df_faiss_indices.mask(mask) # create a new DataFrame with the values excluding excluded elements
            
            if len(df_faiss_indices) != len(df_faiss_indices_with_NaN): # check if length of df and result is the same
                df_faiss_indices_with_NaN = df_faiss_indices_with_NaN.reindex(df_faiss_indices.index)
            
            df_faiss_indices_with_NaN.iloc[:,0] = np.NaN
            df_mask_boolean = df_faiss_indices_with_NaN.iloc[:,:].fillna(0)
            df_mask_boolean = df_mask_boolean.applymap(lambda x: 0 if x == 0 else 1) # df_mask_true_false = df_mask_boolean.applymap(lambda x: False if x == 0 else True)
            

            df_faiss_distances_with_NaN = df_faiss_distances * df_mask_boolean
            df_faiss_distances_with_NaN = df_faiss_distances_with_NaN.replace(0, np.NaN)        

            df_faiss_indices_with_NaN = df_faiss_indices_with_NaN[df_faiss_indices_with_NaN.index.isin(label_samples_id)]
            df_faiss_distances_with_NaN = df_faiss_distances_with_NaN[df_faiss_distances_with_NaN.index.isin(label_samples_id)]


            df_faiss_indices_with_NaN['closest_value'] = df_faiss_indices_with_NaN.apply(closest_value, axis=1)
            df_faiss_distances_with_NaN['closest_value'] = df_faiss_distances_with_NaN.apply(closest_value, axis=1)

            df_faiss_indices_with_NaN_result = df_faiss_indices_with_NaN.loc[:,'closest_value']
            
            sample_selected_result = df_faiss_indices_with_NaN_result[df_faiss_distances_with_NaN.loc[:,'closest_value'].idxmax()]
            sample_selected_result = int(sample_selected_result)
            

            #Add the sample_id in label_sample_ids and remove from unlabeled_sample_ids                
            unlabel_samples_id = np.delete(unlabel_samples_id, np.where(unlabel_samples_id == sample_selected_result))
            label_samples_id = np.concatenate([label_samples_id , np.array([sample_selected_result])])                
                                         
            #Add to the selected_sample_id
            selected_sample_id.append(sample_selected_result)
            pbar.update(1)


    ordered_selected_samples_id = _cold_start_samples_id + selected_sample_id

    return ordered_selected_samples_id


    

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