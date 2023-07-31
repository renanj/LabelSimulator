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

	VAL_SPLIT = [0.20, 0.20] #teste 1 e 2
	# VAL_SPLIT = [0.20] #teste 3

	MAX_SIZE_SPLIT = [625, 1250]  #teste 1
	# MAX_SIZE_SPLIT = [6250, 6250] #teste 2
	# MAX_SIZE_SPLIT = [250] #teste 3

	MAX SIZE = #total do dataset, antes de sofrer o split


	dim_reduction_perplexity = [25, 75]

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
		# '_05_simulations.py': True,
		# '_05_01_building_blocks.py': True,
		# '_05_02_active_learning.py': True,
		# '_06_framework.py': False,
		'_05_framework.py': True,		
		'_06_generate_visualization.py': True,		
		'_07_results_consolidation.py': True		
	}	

	_scripts_order = ['_01_build_dataset.py','_02_feature_extractor.py','_03_dim_reduction.py','_04_generator_faiss.py','_05_framework.py','_05_01_building_blocks.py','_05_02_active_learning.py','_06_generate_visualization.py', '07_results_consolidation.py']

	_files_generated = {
		'_01_build_dataset.py': [
			'df_index_paths_train.pkl', 'df_index_paths_validation.pkl'],
		'_02_feature_extractor.py': [
			'df_train.pkl', 'df_validation.pkl', 'label_encoder.pkl'],
		'_03_dim_reduction.py': [
			'df_2D_train.pkl','df_2D_validation.pkl'],
		'_04_generator_faiss.py': [
			'df_faiss_indices_train.pkl', 'df_faiss_indices_validation.pkl', 'df_faiss_distances_train.pkl', 'df_faiss_distances_validation.pkl',
			'df_2D_faiss_indices_train.pkl', 'df_2D_faiss_indices_validation.pkl', 'df_2D_faiss_distances_train.pkl', 'df_2D_faiss_distances_validation.pkl'
			],

		'_05_framework.py': [
			'df_framework.pkl'
			],

		'_06_generate_visualization.py': [
			'vis_01_consolidate_accuracy_chart_train.png', 'vis_01_consolidate_accuracy_chart_validation.png', 
			'vis_02_accuracy_vs_random_chart_train.png', 'vis_02_accuracy_vs_random_chart_validation.png',
			'vis_04_selection_2Dtrain.png', 'vis_04_selection_2Dvalidation.png'
			],

		'07_results_consolidation.py': [

		]

		# '_05_simulations.py': [
		# 	'df_simulation_samples_ordered_train.pkl', 'df_simulation_samples_ordered_validation.pkl', 'df_simulation_ordered_train.pkl', 'df_simulation_ordered_validation.pkl'],
		# '_05_01_building_blocks.py': [
		# 								],
		# '_05_02_active_learning.py': [
		# 								],
		# '_06_framework.py': [
		# 					'df_framework_train.pkl', 'df_framework_validation.pkl', 'df_simulation_train.pkl', 'df_simulation_validation.pkl', 'vis_accuracy_chart_train.png', 'vis_accuracy_chart_validation.png', 'vis_2D_selection_train.png', 'vis_2D_selection_val.png',  'vis_2D_selection_train.gif', 'vis_2D_selection_val.gif'
		# 					],
		# '_07_results_consolidation.py': [
		# 									]
	}		



	# _list_simulation_sample_name = ['Random', 'NSS', 'SPB','DEN', 'OUT']
	# _list_train_val = ['train'] #, 'val'
	#_list_train_val = ['val'] #, 'val'
	_list_train_val = ['train', 'validation'] #, 'val'
	#controller: 
	_list_data_sets_path = [   


		#teste 1:
		[
			 "data/mnist_1",			 
			 "../../../../../data_colab/mnist/raw",  ## Apontar o RAW para code_lab
			 "data/mnist_1/splited/train",
			 "data/mnist_1/splited/validation",
			 "data/mnist_1/db_feature_extractor",
			 "data/mnist_1/results_consolidated"
		]
		,[
			 "data/plancton_1",
			 "../../../../../data_colab/plancton/raw",  ## Apontar o RAW para code_lab
			 "data/plancton_1/splited/train",
			 "data/plancton_1/splited/validation",
			 "data/plancton_1/db_feature_extractor",
			 "data/plancton_1/results_consolidated"
		]								   



		#teste 2:
		,[
			 "data/mnist_2",			 
			 "../../../../../data_colab/mnist/raw",  ## Apontar o RAW para code_lab
			 "data/mnist_2/splited/train",
			 "data/mnist_2/splited/validation",
			 "data/mnist_2/db_feature_extractor",
			 "data/mnist_2/results_consolidated"
		]
		,[
			 "data/plancton_2",
			 "../../../../../data_colab/plancton/raw",  ## Apontar o RAW para code_lab
			 "data/plancton_2/splited/train",
			 "data/plancton_2/splited/validation",
			 "data/plancton_2/db_feature_extractor",
			 "data/plancton_2/results_consolidated"
		]	


		#teste 3:
		# [
		# 	"data/toy_example",						 
		# 	"data/toy_example/raw",
		# 	"data/toy_example/splited/train",
		# 	"data/toy_example/splited/validation",
		# 	"data/toy_example/db_feature_extractor",
		# 	"data/toy_example/results_consolidated"
		# ]



	]