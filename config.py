# from torchvision import models, transforms

class config:
	# # specify path to the flowers and mnist dataset
	# PLANCTON_DATASET_PATH = "data/plancton/raw"

	# # specify the paths to our training and validation set 
	# PLANCTON_TRAIN = "data/plancton/splited/train"
	# PLANCTON_VAL = "data/plancton/splited/val"

	# set the input height and width
	INPUT_HEIGHT = 128
	INPUT_WIDTH = 128

	# set the batch size and validation data split
	BATCH_SIZE = 0
	VAL_SPLIT = 0.98

	_logs_path = [ "logs"]

	# 0: RAW
	# 1: SPLITED-TRAIN
	# 2: SPLITED-TEST		
	# 3: DB


	_GPU_Flag_dict = {
		'_01_build_dataset.py': True,
		'_02_feature_extractor.py': True,
		'_03_dim_reduction.py': True,
		'_04_generator_faiss.py': True,
		'_05_simulations.py': True,
		'_05_01_building_blocks.py': True,
		'_05_02_active_learning.py': True,
		'_06_framework.py': False,
		'_07_results_consolidation.py': False		
	}	

	_scripts_order = ['_01_build_dataset.py','_02_feature_extractor.py','_03_dim_reduction.py','_04_generator_faiss.py','_05_simulations.py','_05_01_building_blocks.py','_05_02_active_learning.py','_06_framework.py','_07_results_consolidation.py']

	_files_generated = {
		'_01_build_dataset.py': [
			'df_index_paths_train.pkl', 'df_index_paths_validation.pkl'],
		'_02_feature_extractor.py': [
			'df_train.pkl', 'df_validation.pkl'],
		'_03_dim_reduction.py': [
			'df_train.pkl','df_train.pkl'],
		'_04_generator_faiss.py': [
			'df_faiss_indices_train.pkl', 'df_faiss_indices_validation.pkl', 'df_faiss_distances_train.pkl', 'df_faiss_distances_validation.pkl'],
		'_05_simulations.py': [
			'df_simulation_samples_ordered_train.pkl', 'df_simulation_samples_ordered_validation.pkl', 'df_simulation_ordered_train.pkl', 'df_simulation_ordered_validation.pkl'],
		'_05_01_building_blocks.py': [
										],
		'_05_02_active_learning.py': [
										],
		'_06_framework.py': [
							'df_framework_train.pkl', 'df_framework_validation.pkl', 'df_simulation_train.pkl', 'df_simulation_validation.pkl', 'vis_accuracy_chart_train.png', 'vis_accuracy_chart_validation.png', 'vis_2D_selection_train.png', 'vis_2D_selection_val.png',  'vis_2D_selection_train.gif', 'vis_2D_selection_val.gif'
							],
		'_07_results_consolidation.py': [
											]
	}		




	_list_simulation_sample_name = ['Random', 'NSS', 'SPB','DEN', 'OUT']
	# _list_train_val = ['train'] #, 'val'
	#_list_train_val = ['val'] #, 'val'
	_list_train_val = ['train', 'validation'] #, 'val'
	#controller: 
	_list_data_sets_path = [   


		# [
		# 	"data/toy_example",						 
		# 	"data/toy_example/raw",
		# 	"data/toy_example/splited/train",
		# 	"data/toy_example/splited/validation",
		# 	"data/toy_example/db_feature_extractor",
		# 	"data/toy_example/results_consolidated"
		# ]


		[
			 "data/mnist",			 
			 "../../../../../data_colab/mnist/raw",  ## Apontar o RAW para code_lab
			 "data/mnist/splited/train",
			 "data/mnist/splited/validation",
			 "data/mnist/db_feature_extractor",
			 "data/mnist/results_consolidated"
		]



		,[
			 "data/plancton",
			 "../../../../../data_colab/plancton/raw",  ## Apontar o RAW para code_lab
			 "data/plancton/splited/train",
			 "data/plancton/splited/validation",
			 "data/plancton/db_feature_extractor",
			 "data/plancton/results_consolidated"
		]								   


	]
