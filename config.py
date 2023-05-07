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
    VAL_SPLIT = 0.992

    _logs_path = [ "logs"]

    # 0: RAW
    # 1: SPLITED-TRAIN
    # 2: SPLITED-TEST        
    # 3: DB


    _GPU_Flag_dict = {
        '01_build_dataset.py': True,
        '02_feature_extractor.py': True,
        '03_dim_reduction.py': True,
        '04_generator_faiss.py': True,
        '05_simulations.py': True,
        '05_01_building_blocks.py': True,
        '05_02_active_learning.py': True,
        '06_framework.py': False,
        '07_results_consolidation.py': False        
    }    

    _scripts_order = ['01_build_dataset.py','02_feature_extractor.py','03_dim_reduction.py','04_generator_faiss.py','05_simulations.py','05_01_building_blocks.py','05_02_active_learning.py','06_framework.py','07_results_consolidation.py']

    _files_generated = {
        '01_build_dataset.py': [
            'df_index_paths_train.pkl', 'df_index_paths_val.pkl'],
        '02_feature_extractor.py': [
            'df_train.pkl', 'df_train.pkl'],
        '03_dim_reduction.py': [
            'df_train.pkl','df_train.pkl'],
        '04_generator_faiss.py': [
            'df_faiss_indices_train.pkl', 'df_faiss_indices_val.pkl', 'df_faiss_distances_train.pkl', 'df_faiss_distances_val.pkl'],
        '05_simulations.py': [
            'df_simulation_samples_ordered_train.pkl', 'df_simulation_samples_ordered_val.pkl', 'df_simulation_ordered_train.pkl', 'df_simulation_ordered_val.pkl'],
        '05_01_building_blocks.py': [
                                        ],
        '05_02_active_learning.py': [
                                        ],
        '06_framework.py': [
                            'df_framework_train.pkl', 'df_framework_val.pkl', 'df_simulation_train.pkl', 'df_simulation_val.pkl'
                            ],
        '07_results_consolidation.py': [
                                            ]
    }        


    _list_simulation_sample_name = ['Random', 'NSS', 'SPB','DEN', 'OUT']
    _list_train_val = ['train'] #, 'val'
    #_list_train_val = ['val'] #, 'val'
    #controller: 
    _list_data_sets_path = [   


        # [
        #     "data/toy_example",                         
        #     "data/toy_example/raw",
        #     "data/toy_example/splited/train",
        #     "data/toy_example/splited/val",
        #     "data/toy_example/db_feature_extractor",
        #     "data/toy_example/results_consolidated"
        # ]


        [
            "data/mnist",             
            "../../../../../data_colab/mnist/raw",  ## Apontar o RAW para code_lab
            "data/mnist/splited/train",
            "data/mnist/splited/val",
            "data/mnist/db_feature_extractor",
            "data/mnist/results_consolidated"
        ]



        # ,[
        #     "data/plancton",
        #     "../../../../../data_colab/plancton/raw",  ## Apontar o RAW para code_lab
        #     "data/plancton/splited/train",
        #     "data/plancton/splited/val",
        #     "data/plancton/db_feature_extractor",
        #     "data/plancton/results_consolidated"
        # ]                                   


    ]
