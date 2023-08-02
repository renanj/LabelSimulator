import os

class Config:

	def __init__(self,test_number):

		self._colab_backup_path = '../../../../../backup_LabelSimulator/'
		self._run_colab_backup_path = True

		self.test_number = test_number
		self._list_train_val = ['train', 'validation']

		self._GPU_Flag_dict = {
			'_01_build_dataset.py': True,
			'_02_feature_extractor.py': True,
			'_03_dim_reduction.py': True,
			'_04_generator_faiss.py': True,
			'_human_simulations_files.py': True,
			'_05_01_building_blocks.py': True,
			'_05_02_active_learning.py': True,
			'_05_framework.py': True,		
			'_06_generate_visualization.py': True,		
			'_07_results_consolidation.py': True		
		}	


		self._scripts_order = [	
		'_01_build_dataset.py',
		'_02_feature_extractor.py',
		'_03_dim_reduction.py',
		'_04_generator_faiss.py',
		'_human_simulations_files.py',
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
		'_human_simulations_files.py': ['df_simulation_order_df.pkl', 'df_simulation_order_df_2D.pkl'],		
		'_05_framework.py': ['df_framework.pkl', 'df_framework_temporary'],
		'_06_generate_visualization.py': ['vis_01_consolidate_accuracy_chart_train.png', 'vis_01_consolidate_accuracy_chart_validation.png', 'vis_02_accuracy_vs_random_chart_train.png', 'vis_02_accuracy_vs_random_chart_validation.png','vis_04_selection_2Dtrain.png', 'vis_04_selection_2Dvalidation.png'],
		'07_results_consolidation.py': []
		}


		
		#Lista de Modelos que queremos que rode com Ensemble de Regressao Logistica (mais demorado) 
		self._ensembles_heuristics_list = ['Bald', 'BatchBALD', 'PowerBALD'] 


		if self.test_number == 'toy_example':

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


			self._list_query_stragegy = ['Random', 
										'Uncertainty', 'Margin', 'Entropy', 'Bald', 'BatchBALD', #PowerBALD,
										'Equal_Spread', 'Dense_Areas_First', 'Centroids_First',  'Outliers_First', 
										'Equal_Spread_2D', 'Dense_Areas_First_2D', 'Centroids_First_2D',  'Outliers_First_2D']	

			self.human_simulations = True #Manually Check if the _list_query_strategies contain any of this...
			self.load_human_simulations_files = False #Just use if we saved the files in google driver -- once they take a lot of time to run and the process may failed															
			self._batch_size_experiment = True
			self._batch_size_options = [10, 25, 50]		
			self._list_strategies_for_batch_size_comparison = ['Random','Uncertainty', 'Margin', 'Entropy', 'Bald', 'BatchBALD'] #PowerBALD	


		elif self.test_number == 'mnist_1':

			self.VAL_SPLIT = [0.20]
			self.MAX_SIZE_SPLIT = [625]

			self._list_data_sets_path = [   
				[
					 "data/mnist_1",			 
					 "../../../../../data_colab/mnist/raw",
					 "data/mnist_1/splited/train",
					 "data/mnist_1/splited/validation",
					 "data/mnist_1/db_feature_extractor",
					 "data/mnist_1/results_consolidated"
				]			
			]

			self._list_query_stragegy = ['Random', 
										'Uncertainty', 'Margin', 'Entropy', 'Bald', 'BatchBALD', #PowerBALD,
										'Equal_Spread', 'Dense_Areas_First', 'Centroids_First',  'Outliers_First', 
										'Equal_Spread_2D', 'Dense_Areas_First_2D', 'Centroids_First_2D',  'Outliers_First_2D']	

			self.human_simulations = True #Manually Check if the _list_query_strategies contain any of this...
			self.load_human_simulations_files = False #Just use if we saved the files in google driver -- once they take a lot of time to run and the process may failed												
			
			self._batch_size_experiment = True
			self._batch_size_options = [1, 5, 10, 25, 50, 100]			
			self._list_strategies_for_batch_size_comparison = ['Random','Uncertainty', 'Margin', 'Entropy', 'Bald', 'BatchBALD'] #PowerBALD	



		elif self.test_number == 'mnist_2':

			self.VAL_SPLIT = [0.20]
			self.MAX_SIZE_SPLIT = [6250]	

			self._list_data_sets_path = [   
				[
					 "data/mnist_2",			 
					 "../../../../../data_colab/mnist/raw",
					 "data/mnist_2/splited/train",
					 "data/mnist_2/splited/validation",
					 "data/mnist_2/db_feature_extractor",
					 "data/mnist_2/results_consolidated"
				]			
			]


			self._list_query_stragegy = ['Random', 
										'Uncertainty', 'Margin', 'Entropy', 'Bald', 'BatchBALD', #PowerBALD,
										'Equal_Spread', 'Dense_Areas_First', 'Centroids_First',  'Outliers_First', 
										'Equal_Spread_2D', 'Dense_Areas_First_2D', 'Centroids_First_2D',  'Outliers_First_2D']	

			self.human_simulations = True #Manually Check if the _list_query_strategies contain any of this...
			self.load_human_simulations_files = False #Just use if we saved the files in google driver -- once they take a lot of time to run and the process may failed												
			
			self._batch_size_experiment = True
			self._batch_size_options = [10, 25, 50, 100, 500, 1000]			
			self._list_strategies_for_batch_size_comparison = ['Random','Uncertainty', 'Margin', 'Entropy', 'Bald', 'BatchBALD'] #PowerBALD		
	

		elif self.test_number == 'plancton_1':

			self.VAL_SPLIT = [0.20]
			self.MAX_SIZE_SPLIT = [1250]

			self._list_data_sets_path = [   
				[
					 "data/plancton_1",			 
					 "../../../../../data_colab/plancton/raw",
					 "data/plancton_1/splited/train",
					 "data/plancton_1/splited/validation",
					 "data/plancton_1/db_feature_extractor",
					 "data/plancton_1/results_consolidated"
				]				
			]

			self._list_query_stragegy = ['Random', 
										'Uncertainty', 'Margin', 'Entropy', 'Bald', 'BatchBALD', #PowerBALD,
										'Equal_Spread', 'Dense_Areas_First', 'Centroids_First',  'Outliers_First', 
										'Equal_Spread_2D', 'Dense_Areas_First_2D', 'Centroids_First_2D',  'Outliers_First_2D']	

			self.human_simulations = True #Manually Check if the _list_query_strategies contain any of this...
			self.load_human_simulations_files = False #Just use if we saved the files in google driver -- once they take a lot of time to run and the process may failed												
			
			self._batch_size_experiment = True
			self._batch_size_options = [1, 5, 10, 25, 50, 100]			
			self._list_strategies_for_batch_size_comparison = ['Random','Uncertainty', 'Margin', 'Entropy', 'Bald', 'BatchBALD'] #PowerBALD					


		elif self.test_number == 'plancton_2':			

			self.VAL_SPLIT = [0.20]
			self.MAX_SIZE_SPLIT = [6250]	

			self._list_data_sets_path = [   
				[
					 "data/plancton_2",			 
					 "../../../../../data_colab/plancton/raw",
					 "data/plancton_2/splited/train",
					 "data/plancton_2/splited/validation",
					 "data/plancton_2/db_feature_extractor",
					 "data/plancton_2/results_consolidated"
				]				
			]


			self._list_query_stragegy = ['Random', 
										'Uncertainty', 'Margin', 'Entropy', 'Bald', 'BatchBALD', #PowerBALD,
										'Equal_Spread', 'Dense_Areas_First', 'Centroids_First',  'Outliers_First', 
										'Equal_Spread_2D', 'Dense_Areas_First_2D', 'Centroids_First_2D',  'Outliers_First_2D']	

			self.human_simulations = True #Manually Check if the _list_query_strategies contain any of this...
			self.load_human_simulations_files = False #Just use if we saved the files in google driver -- once they take a lot of time to run and the process may failed												
			
			self._batch_size_experiment = True
			self._batch_size_options = [10, 25, 50, 100, 500, 1000]			
			self._list_strategies_for_batch_size_comparison = ['Random','Uncertainty', 'Margin', 'Entropy', 'Bald', 'BatchBALD'] #PowerBALD		

		else:
			raise ValueError("Invalid configuration name!")						

	
# Get the configuration name from the environment variable.
test_number = os.getenv('TEST_NUMBER')

if test_number is None:
    raise ValueError('TEST_NUMBER environment variable not set. Please provide a configuration name.')

# Create a Config instance with the appropriate configuration.
config = Config(test_number)