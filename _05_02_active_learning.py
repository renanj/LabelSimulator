import pandas as pd
import numpy as np

import random
import math
import warnings
import time
warnings.filterwarnings('ignore')
# import faiss
# from faiss import StandardGpuResources
from tqdm import tqdm
import cupy as cp
import cudf

from _05_01_building_blocks import f_cold_start

from baal.active.heuristics import BALD, Certainty, Margin, Entropy, Variance, Random, BatchBALD
from sklearn.linear_model import LogisticRegression

from aux_functions import f_model_accuracy







def f_run_active_learning(df_embbedings, _df_validation, _model=None, _query_size=1, _al_query_list=None, _cold_start_samples_id=None, _model_name=None):


	if _al_query_list is None:
		_al_query_list = ['Uncertainty', 'Entropy', 'Margin', 'Bald'] #, 'BatchBald'


	if _cold_start_samples_id is None: 
		_samples_id_list_random, _cold_start_samples_id = f_cold_start(df_embbedings)


	if _model is None:
		_model_1 = LogisticRegression(random_state=0)
		_model_2 = LogisticRegression(random_state=0)
		_model_3 = LogisticRegression(random_state=0)
		_model_4 = LogisticRegression(random_state=0)
		# _model_5 = LogisticRegression(random_state=0)



	_temp_X_columns = [x for x, mask in zip(df_embbedings.columns.values, df_embbedings.columns.str.startswith("X")) if mask]

	X_train = df_embbedings[df_embbedings['sample_id'].isin(_cold_start_samples_id)].loc[:,_temp_X_columns].astype('float32')
	y_train = df_embbedings[df_embbedings['sample_id'].isin(_cold_start_samples_id)].loc[:,'labels'].astype('float32')	   

	X_test = df_embbedings.loc[:,_temp_X_columns].astype('float32')
	y_test = df_embbedings.loc[:,'labels'].astype('float32')

	X_validation = _df_validation.loc[:,_temp_X_columns].astype('float32')
	y_validation = _df_validation.loc[:,'labels'].astype('float32')			 

   
	_model_1.fit(X_train, y_train)
	_model_2.fit(X_train, y_train)
	_model_3.fit(X_train, y_train)
	_model_4.fit(X_train, y_train)
	# _model_5.fit(X_train, y_train)  

	_array_score_validation_1 = cp.array(_model_1.score(X_validation, y_validation))
	_array_score_validation_2 = cp.array(_model_2.score(X_validation, y_validation))
	_array_score_validation_3 = cp.array(_model_3.score(X_validation, y_validation))
	_array_score_validation_4 = cp.array(_model_4.score(X_validation, y_validation))
	# _array_score_validation_5 = cp.array(_model_5.score(X_validation, y_validation))



	_array_labels_sample_ids_1 = cp.array(_cold_start_samples_id)
	_array_unlabels_sample_ids_1 = cp.array(df_embbedings['sample_id'][~df_embbedings['sample_id'].isin(_cold_start_samples_id)])

	_array_labels_sample_ids_2 = cp.array(_cold_start_samples_id)
	_array_unlabels_sample_ids_2 = cp.array(df_embbedings['sample_id'][~df_embbedings['sample_id'].isin(_cold_start_samples_id)])

	_array_labels_sample_ids_3 = cp.array(_cold_start_samples_id)
	_array_unlabels_sample_ids_3 = cp.array(df_embbedings['sample_id'][~df_embbedings['sample_id'].isin(_cold_start_samples_id)])

	_array_labels_sample_ids_4 = cp.array(_cold_start_samples_id)
	_array_unlabels_sample_ids_4 = cp.array(df_embbedings['sample_id'][~df_embbedings['sample_id'].isin(_cold_start_samples_id)])

	# _array_labels_sample_ids_5 = cp.array(_cold_start_samples_id)
	# _array_unlabels_sample_ids_5 = cp.array(df_embbedings['sample_id'][~df_embbedings['sample_id'].isin(_cold_start_samples_id)])


	start_time_while = time.time()
	
	while len(_array_unlabels_sample_ids_4) > 0:  


		print("Missing = ", len(_array_unlabels_sample_ids_4))
		


		#Model 1:
		print("Model 1 = Uncertainty")
		start_time = time.time()
		_temp_test_x = df_embbedings[_temp_X_columns][df_embbedings['sample_id'].isin(_array_unlabels_sample_ids_1.get())] #AQUI
		x = _model_1.predict_proba(_temp_test_x)
		x = x.reshape(x.shape[0], x.shape[1], 1)		

		_baal_scores = Certainty().compute_score(x) #AQUI
		_baal_rank = Certainty()(x) #AQUI

		selected_sample_id = _array_unlabels_sample_ids_1[_baal_rank[:_query_size]] #AQUI
		_array_unlabels_sample_ids_1 = _array_unlabels_sample_ids_1[_array_unlabels_sample_ids_1 != selected_sample_id] #AQUI
		_array_labels_sample_ids_1= cp.append(_array_labels_sample_ids_1, selected_sample_id) #AQUI		 


		#re-train: 		
		X_train = df_embbedings[df_embbedings['sample_id'].isin(_array_labels_sample_ids_1.get())].loc[:,_temp_X_columns].astype('float32') #AQUI
		y_train = df_embbedings[df_embbedings['sample_id'].isin(_array_labels_sample_ids_1.get())].loc[:,'labels'].astype('float32') #AQUI	   
		_model_1.fit(X_train, y_train) #AQUI	

		_score_validation = cp.array(_model_1.score(X_validation, y_validation)) #AQUI
		_array_score_validation_1 = cp.append(_array_score_validation_1, _score_validation) #AQUI

		end_time = time.time()
		execution_time = end_time - start_time
		print(f"Execution time: {execution_time} seconds")		


		#Model 2:
		print("Model 2 = Entropy")
		start_time = time.time()
		_temp_test_x = df_embbedings[_temp_X_columns][df_embbedings['sample_id'].isin(_array_unlabels_sample_ids_2.get())] #AQUI
		x = _model_2.predict_proba(_temp_test_x)
		x = x.reshape(x.shape[0], x.shape[1], 1)		

		_baal_scores = Entropy().compute_score(x) #AQUI
		_baal_rank = Entropy()(x) #AQUI

		selected_sample_id = _array_unlabels_sample_ids_2[_baal_rank[:_query_size]] #AQUI
		_array_unlabels_sample_ids_2 = _array_unlabels_sample_ids_2[_array_unlabels_sample_ids_2 != selected_sample_id] #AQUI
		_array_labels_sample_ids_2= cp.append(_array_labels_sample_ids_2, selected_sample_id) #AQUI		 


		#re-train: 		
		X_train = df_embbedings[df_embbedings['sample_id'].isin(_array_labels_sample_ids_2.get())].loc[:,_temp_X_columns].astype('float32') #AQUI
		y_train = df_embbedings[df_embbedings['sample_id'].isin(_array_labels_sample_ids_2.get())].loc[:,'labels'].astype('float32') #AQUI	   
		_model_2.fit(X_train, y_train) #AQUI	

		_score_validation = cp.array(_model_2.score(X_validation, y_validation)) #AQUI
		_array_score_validation_2 = cp.append(_array_score_validation_2, _score_validation) #AQUI

		end_time = time.time()
		execution_time = end_time - start_time
		print(f"Execution time: {execution_time} seconds")		



		#Model 3:
		print("Model 3 = Margin")
		start_time = time.time()
		_temp_test_x = df_embbedings[_temp_X_columns][df_embbedings['sample_id'].isin(_array_unlabels_sample_ids_3.get())] #AQUI
		x = _model_3.predict_proba(_temp_test_x)
		x = x.reshape(x.shape[0], x.shape[1], 1)		

		_baal_scores = Margin().compute_score(x) #AQUI
		_baal_rank = Margin()(x) #AQUI

		selected_sample_id = _array_unlabels_sample_ids_3[_baal_rank[:_query_size]] #AQUI
		_array_unlabels_sample_ids_3 = _array_unlabels_sample_ids_3[_array_unlabels_sample_ids_3 != selected_sample_id] #AQUI
		_array_labels_sample_ids_3= cp.append(_array_labels_sample_ids_3, selected_sample_id) #AQUI		 


		#re-train: 		
		X_train = df_embbedings[df_embbedings['sample_id'].isin(_array_labels_sample_ids_3.get())].loc[:,_temp_X_columns].astype('float32') #AQUI
		y_train = df_embbedings[df_embbedings['sample_id'].isin(_array_labels_sample_ids_3.get())].loc[:,'labels'].astype('float32') #AQUI	   
		_model_3.fit(X_train, y_train) #AQUI	

		_score_validation = cp.array(_model_3.score(X_validation, y_validation)) #AQUI
		_array_score_validation_3 = cp.append(_array_score_validation_3, _score_validation) #AQUI

		end_time = time.time()
		execution_time = end_time - start_time
		print(f"Execution time: {execution_time} seconds")		



		#Model 4:
		print("Model 4 = BALD")
		start_time = time.time()
		_temp_test_x = df_embbedings[_temp_X_columns][df_embbedings['sample_id'].isin(_array_unlabels_sample_ids_4.get())] #AQUI
		x = _model_4.predict_proba(_temp_test_x)
		x = x.reshape(x.shape[0], x.shape[1], 1)		

		_baal_scores = BALD().compute_score(x) #AQUI
		_baal_rank = BALD()(x) #AQUI

		selected_sample_id = _array_unlabels_sample_ids_4[_baal_rank[:_query_size]] #AQUI
		_array_unlabels_sample_ids_4 = _array_unlabels_sample_ids_4[_array_unlabels_sample_ids_4 != selected_sample_id] #AQUI
		_array_labels_sample_ids_4= cp.append(_array_labels_sample_ids_4, selected_sample_id) #AQUI		 


		#re-train: 		
		X_train = df_embbedings[df_embbedings['sample_id'].isin(_array_labels_sample_ids_4.get())].loc[:,_temp_X_columns].astype('float32') #AQUI
		y_train = df_embbedings[df_embbedings['sample_id'].isin(_array_labels_sample_ids_4.get())].loc[:,'labels'].astype('float32') #AQUI	   
		_model_4.fit(X_train, y_train) #AQUI	

		_score_validation = cp.array(_model_4.score(X_validation, y_validation)) #AQUI
		_array_score_validation_4 = cp.append(_array_score_validation_4, _score_validation) #AQUI		

		end_time = time.time()
		execution_time = end_time - start_time
		print(f"Execution time: {execution_time} seconds")		
		print("---------------\n\n\n\n")



		# #Model 5:
		# _temp_test_x = df_embbedings[_temp_X_columns][df_embbedings['sample_id'].isin(_array_unlabels_sample_ids_5.get())] #AQUI
		# x = _model_5.predict_proba(_temp_test_x)
		# x = x.reshape(x.shape[0], x.shape[1], 1)		

		# _baal_scores = BatchBALD(num_samples=10).compute_score(x) #AQUI
		# _baal_rank = BatchBALD(num_samples=10)(x) #AQUI

		# selected_sample_id = _array_unlabels_sample_ids_5[_baal_rank[:_query_size]] #AQUI
		# _array_unlabels_sample_ids_5 = _array_unlabels_sample_ids_5[_array_unlabels_sample_ids_5 != selected_sample_id] #AQUI
		# _array_labels_sample_ids_5= cp.append(_array_labels_sample_ids_5, selected_sample_id) #AQUI		 


		# #re-train: 		
		# X_train = df_embbedings[df_embbedings['sample_id'].isin(_array_labels_sample_ids_5.get())].loc[:,_temp_X_columns].astype('float32') #AQUI
		# y_train = df_embbedings[df_embbedings['sample_id'].isin(_array_labels_sample_ids_5.get())].loc[:,'labels'].astype('float32') #AQUI	   
		# _model_5.fit(X_train, y_train) #AQUI	

		# _score_validation = cp.array(_model_5.score(X_validation, y_validation)) #AQUI
		# _array_score_validation_5 = cp.append(_array_score_validation_5, _score_validation) #AQUI			


	end_time_while = time.time()
	execution_time = end_time_while - start_time_while
	print(f"Execution time: {execution_time} seconds")

	_list_simulations_sample_id = []
	_list_simulations_proceeded = []


	_list_simulations_sample_id.append(_array_labels_sample_ids_1.tolist())
	_list_simulations_sample_id.append(_array_labels_sample_ids_2.tolist())
	_list_simulations_sample_id.append(_array_labels_sample_ids_3.tolist())
	_list_simulations_sample_id.append(_array_labels_sample_ids_4.tolist())
	# _list_simulations_sample_id.append(_array_labels_sample_ids_5.tolist())

	_list_simulations_proceeded.append(_al_query_list[0])
	_list_simulations_proceeded.append(_al_query_list[1])
	_list_simulations_proceeded.append(_al_query_list[2])
	_list_simulations_proceeded.append(_al_query_list[3])
	# _list_simulations_proceeded.append(_al_query_list[4])


	return _list_simulations_proceeded, _list_simulations_sample_id