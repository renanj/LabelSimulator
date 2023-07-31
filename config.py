import os os

class Config:

	def __init__(self,test_number):

		self.test_number = test_number
		self._list_train_val = ['train', 'validation']

		self._GPU_Flag_dict = {
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


		self._scripts_order = [	
		'_01_build_dataset.py',
		'_02_feature_extractor.py',
		'_03_dim_reduction.py',
		'_04_generator_faiss.py',
		'_05_framework.py',
		'_05_01_building_blocks.py',
		'_05_02_active_learning.py',
		'_06_generate_visualization.py', 
		'07_results_consolidation.py'
		]


		self._files_generated = {
		'_01_build_dataset.py': ['df_index_paths_train.pkl', 'df_index_paths_validation.pkl'],
		'_02_feature_extractor.py': ['df_train.pkl', 'df_validation.pkl', 'label_encoder.pkl'],
		'_03_dim_reduction.py': ['df_2D_train.pkl','df_2D_validation.pkl'],
		'_04_generator_faiss.py': ['df_faiss_indices_train.pkl', 'df_faiss_indices_validation.pkl', 'df_faiss_distances_train.pkl', 'df_faiss_distances_validation.pkl','df_2D_faiss_indices_train.pkl', 'df_2D_faiss_indices_validation.pkl', 'df_2D_faiss_distances_train.pkl', 'df_2D_faiss_distances_validation.pkl'],
		'_05_framework.py': ['df_framework.pkl'],
		'_06_generate_visualization.py': ['vis_01_consolidate_accuracy_chart_train.png', 'vis_01_consolidate_accuracy_chart_validation.png', 'vis_02_accuracy_vs_random_chart_train.png', 'vis_02_accuracy_vs_random_chart_validation.png','vis_04_selection_2Dtrain.png', 'vis_04_selection_2Dvalidation.png'],
		'07_results_consolidation.py': []
		}


		if self.test_number == 'teste_1':

			self.VAL_SPLIT = [0.20, 0.20]
			self.MAX_SIZE_SPLIT = [625, 1250]			
			self._list_data_sets_path = [   
				[
					 "data/mnist_1",			 
					 "../../../../../data_colab/mnist/raw",
					 "data/mnist_1/splited/train",
					 "data/mnist_1/splited/validation",
					 "data/mnist_1/db_feature_extractor",
					 "data/mnist_1/results_consolidated"
				],
				[
					 "data/plancton_1",			 
					 "../../../../../data_colab/plancton/raw",
					 "data/plancton_1/splited/train",
					 "data/plancton_1/splited/validation",
					 "data/plancton_1/db_feature_extractor",
					 "data/plancton_1/results_consolidated"
				]				
			]
			self._batch_size_experiment = True
			self._batch_size_options = [1, 5, 10, 25, 50, 100]


		elif self.test_number == 'teste_2':

			self.VAL_SPLIT = [0.20, 0.20]
			self.MAX_SIZE_SPLIT = [6250, 6250]			
			self._list_data_sets_path = [   
				[
					 "data/mnist_2",			 
					 "../../../../../data_colab/mnist/raw",
					 "data/mnist_2/splited/train",
					 "data/mnist_2/splited/validation",
					 "data/mnist_2/db_feature_extractor",
					 "data/mnist_2/results_consolidated"
				],
				[
					 "data/plancton_2",			 
					 "../../../../../data_colab/plancton/raw",
					 "data/plancton_2/splited/train",
					 "data/plancton_2/splited/validation",
					 "data/plancton_2/db_feature_extractor",
					 "data/plancton_2/results_consolidated"
				]				
			]
			self._batch_size_experiment = True
			self._batch_size_options = [10, 25, 50, 100, 500, 1000]



		elif self.test_number == 'teste_3':

			self.VAL_SPLIT = [0.20]
			self.MAX_SIZE_SPLIT = [300]			
			self._list_data_sets_path = [   
				[
					"data/toy_example",						 
					"data/toy_example/raw",
					"data/toy_example/splited/train",
					"data/toy_example/splited/validation",
					"data/toy_example/db_feature_extractor",
					"data/toy_example/results_consolidated"
				]			  
			]
			self._batch_size_experiment = True
			self._batch_size_options = [10, 25, 50]			

		else:
			raise ValueError("Invalid configuration name!")
	


	
# Get the configuration name from the environment variable.
test_number = os.getenv('TEST_NUMBER')

if test_number is None:
    raise ValueError('TEST_NUMBER environment variable not set. Please provide a configuration name.')

# Create a Config instance with the appropriate configuration.
config = Config(test_number)
