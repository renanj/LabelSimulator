import pandas as pd
import numpy as np
import random
import itertools
from collections import OrderedDict
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
		_cold_start_samples_id  = _random_samples_id[0:math.ceil(0.05*len(_random_samples_id))]

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
	
	out_ordered_selected_samples_id = desentity_ordered_selected_samples_id.copy()
	out_ordered_selected_samples_id.reverse()
	# print("Before:")
	# print("First 5 = ", desentity_ordered_selected_samples_id[:5])
	# print("Final 5 = ", desentity_ordered_selected_samples_id[:-5])	
	# print("After:")
	# print("First 5 = ", out_ordered_selected_samples_id[:5])
	# print("Final 5 = ", out_ordered_selected_samples_id[:-5])  

	return out_ordered_selected_samples_id



def f_clu(df_embbedings, num_clusters=None, num_iterations=None, gpu_index=True):

	if num_clusters is None:
		num_clusters = round(df_embbedings.shape[0] * 0.025)

	if num_iterations is None:
		num_iterations = 10

	  

	_temp_X_columns = list(df_embbedings.loc[:,df_embbedings.columns.str.startswith("X")].columns)
	if _temp_X_columns == None:
	  _temp_X_columns = list(df_embbedings.loc[:,df_embbedings.columns.str.startswith("X")].columns)


	X = np.ascontiguousarray(df_embbedings[_temp_X_columns].values.astype('float32'))
	d = X.shape[1]


	# 3) Settings the IDs based on sample_id column & creating the correct "index"
	sample_ids = df_embbedings['sample_id'].values  
	index = faiss.IndexFlatL2(d)
	index = faiss.IndexIDMap(index)	 
	index.add_with_ids(X, sample_ids) # index.add(X) #creating the index with the sample_ids instead of sequential value #For reference: https://github.com/facebookresearch/faiss/wiki/Pre--and-post-processing	






	# Run k-means clustering
	# kmeans = faiss.Kmeans(X.shape[1], num_clusters, niter=num_iterations, verbose=True)
	kmeans = faiss.Kmeans(X.shape[1], num_clusters, niter=5, verbose=True)  
	# kmeans = faiss.Kmeans(X.shape[1], num_clusters, verbose=True)  
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





def join_lists_equal_distance(_list_a, _list_b):

	if len(_list_a) <= len(_list_b):
		list_1 = _list_a
		list_2 = _list_b
	else:
		list_1 = _list_b
		list_2 = _list_a		


	#To dynamically insert elements from list_1 into list_2 with equal distance between each insert
	list_1 = cp.array(list_1)
	list_2 = cp.array(list_2)
	# Calculate the spacing between each insert
	spacing = len(list_2) // len(list_1) + 1
	# Generate a range of indices to insert the elements from list_1
	indices = cp.arange(spacing, len(list_2), spacing)
	indices_tuple = tuple(map(int, indices))
	# Slice list_2 into segments between the insertion indices
	segments = [cp.array(list_2[i:j]) for i,j in zip([0]+indices.tolist(), indices.tolist()+[None])]
	# Concatenate the segments with list_1
	result = cp.concatenate([cp.concatenate([s, cp.array([l])]) for s,l in zip(segments, list_1)])
	result = result.tolist()
	return result

	



def f_run_human_simulations(df_embbedings, df_faiss_distances, df_faiss_indices, _query_size=1, _query_strategy=None, _cold_start_samples_id=None, _human_simulation_list=None):


	if _human_simulation_list is None:
		# _human_simulation_list = ['Random', 'Equal_Spread', 'Dense_Areas_First', 'Centroids_First', 'Cluster_Boarder_First',  'Outliers_First']
		_human_simulation_list = ['Random', 'Equal_Spread', 'Dense_Areas_First', 'Centroids_First',  'Outliers_First']
		# _human_simulation_list = ['Random', 'Equal_Spread']
	else:
		if 'Random' not in _human_simulation_list:
			_human_simulation_list.append('Random')
		else: 
			None


	#FAISS INDICES & DISTANCES DATAFRAME	
	# df_faiss_indices, df_faiss_distances = bblocks.f_faiss(df_embbedings)

	#BUILDING BLOCKS:
	print("[INFO] -- Creating Building Blocks...")
	print("Random:")
	_samples_id_list_random, _samples_id_list_random_cold_start = f_cold_start(df_embbedings)	
	print("SPB:")
	_samples_id_list_ordered_SPB = f_SPB(df_embbedings, df_faiss_distances, df_faiss_indices, _cold_start_samples_id=_samples_id_list_random_cold_start)
	print("DEN:")
	_samples_id_list_ordered_DEN = f_den(df_embbedings, df_faiss_distances, df_faiss_indices, _cold_start_samples_id=_samples_id_list_random_cold_start, k=5)
	print("OUT:")
	_samples_id_list_ordered_OUT = f_out(_samples_id_list_ordered_DEN)
	print("CLU:")
	_centroids_samples_id_list_ordered_CLU, _clusters_samples_id_list_of_lists_ordered_CLU = f_clu(df_embbedings, num_clusters=None, num_iterations=10, gpu_index=True)
	print("------------------------------------------------\n\n")



	
	_list_simulations_sample_id = []
	_list_simulations_proceeded = []
	print("[INFO] -- Starting Simulation...")

	#SIMULATION RUN based on "_human_simulation_list":
	for _sim in _human_simulation_list:

		if _sim == 'Random':
			print("Starting Random...")			
			_list_simulations_sample_id.append(_samples_id_list_random)
			_list_simulations_proceeded.append(_sim)
			print("Qtd Samples = ", len(_samples_id_list_random))
			print("End Random!")
			print("--------------------\n")


		elif _sim == 'Equal_Spread':
			print("Starting Equal Spread...")						
			_samples_id_ordered = _samples_id_list_ordered_SPB
			_list_simulations_sample_id.append(_samples_id_ordered)
			_list_simulations_proceeded.append(_sim)
			print("Qtd Samples = ", len(_samples_id_ordered))
			print("End Equal Spread!")
			print("--------------------\n")


		elif _sim == 'Dense_Areas_First':
			print("Starting Dense_Areas_First...")	
			print("_samples_id_list_ordered_SPB type =", type(_samples_id_list_ordered_SPB))		
			print("_samples_id_list_ordered_DEN type =", type(_samples_id_list_ordered_DEN))					
			_samples_id_ordered = list(itertools.chain.from_iterable(zip(_samples_id_list_ordered_SPB, _samples_id_list_ordered_DEN)))
			_samples_id_ordered = list(OrderedDict.fromkeys(_samples_id_ordered))
			_list_simulations_sample_id.append(_samples_id_ordered)
			_list_simulations_proceeded.append(_sim)

			print("Qtd Samples = ", len(_samples_id_ordered))
			print("End Dense_Areas_First!")
			print("--------------------\n")


		elif _sim == 'Centroids_First':
			print("Starting Centroids_First...")						
			_samples_id_ordered = join_lists_equal_distance(_samples_id_list_ordered_SPB, _centroids_samples_id_list_ordered_CLU)
			_samples_id_ordered = list(OrderedDict.fromkeys(_samples_id_ordered))		
			_list_simulations_sample_id.append(_samples_id_ordered)
			_list_simulations_proceeded.append(_sim)
			print("Qtd Samples = ", len(_samples_id_ordered))
			print("End Centroids_First!")
			print("--------------------\n")	


		# elif _sim == 'Cluster_Borders_First':
		#	 print("Starting Cluster_Borders_First'...")			
			
		#	 _samples_id_ordered_by_cluster = []
		#	 for i in range(len(_clusters_samples_id_list_of_lists_ordered_CLU)):
		#		 samples_id_in_cluster = _clusters_samples_id_list_of_lists_ordered_CLU[i]

		#		 #SPB:
		#		 df_temp = df_embbedings[df_embbedings['sample_id'].isin(samples_id_in_cluster)].copy()
		#		 df_temp = df_temp.reset_index(drop=True)

		#		 df_faiss_distances_temp = df_faiss_distances[df_faiss_distances.index.isin(samples_id_in_cluster)].copy()
		#		 df_faiss_distances_temp = df_faiss_distances_temp.reset_index(drop=True)

		#		 df_faiss_indices_temp = df_faiss_distances[df_faiss_distances.index.isin(samples_id_in_cluster)].copy()
		#		 df_faiss_indices_temp = df_faiss_indices_temp.reset_index(drop=True)

		#		 _temp_samples_id_list_random, _temp_samples_id_list_random_cold_start = bblocks.f_cold_start(df_temp)	
		#		 _temp_samples_id_list_ordered_SPB = bblocks.f_SPB(df_temp, df_faiss_distances_temp, df_faiss_indices_temp, _cold_start_samples_id=_temp_samples_id_list_random_cold_start)

		#		 #OUT:
		#		 _temp_samples_id_list_ordered_DEN = bblocks.f_den(df_temp, df_faiss_distances_temp, df_faiss_indices_temp, _cold_start_samples_id=_temp_samples_id_list_random_cold_start, k=5)
		#		 _temp_samples_id_list_ordered_OUT = bblocks.f_out(_temp_samples_id_list_ordered_DEN)				


		#		 #SPB(50%) + OUT(50%) within Cluster:
		#		 _temp_list = [val for pair in zip(_temp_samples_id_list_ordered_SPB, _temp_samples_id_list_ordered_OUT) for val in pair]
		#		 _temp_list = list(set(_temp_list))				
		#		 _samples_id_ordered_by_cluster.append(_temp_list)

			
			# _samples_id_ordered = [val for pair in zip(*_samples_id_ordered_by_cluster) for val in pair]
			# print("Len List = ", len(_samples_id_ordered))
			# _samples_id_ordered = list(set(_samples_id_ordered))				
			# print("Len List = ", len(_samples_id_ordered))

			# _list_simulations_sample_id.append(_samples_id_ordered)
			# _list_simulations_proceeded.append(_sim)
			# print("Qtd Samples = ", len(_samples_id_ordered))
			# print("End Cluster_Borders_First'!")
			# print("--------------------")	


		elif _sim == 'Outliers_First':
			print("Starting Outliers_First...")						
			_samples_id_ordered = list(itertools.chain.from_iterable(zip(_samples_id_list_ordered_SPB, _samples_id_list_ordered_OUT)))
			_samples_id_ordered = list(OrderedDict.fromkeys(_samples_id_ordered))
			_list_simulations_sample_id.append(_samples_id_ordered)
			_list_simulations_proceeded.append(_sim)
			print("Qtd Samples = ", len(_samples_id_ordered))
			print("End Outliers_First!")
			print("--------------------\n")

		
		else:
			print("We don't have a function ready for {} simulation!", _sim)
	return _list_simulations_proceeded, _list_simulations_sample_id

