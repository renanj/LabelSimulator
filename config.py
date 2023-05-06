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
    VAL_SPLIT = 0.9

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


    

    _list_simulation_sample_name = ['Random', 'NSS', 'SPB','DEN', 'OUT']
    _list_train_val = ['train'] #, 'val'
    #_list_train_val = ['val'] #, 'val'
    #controller: 
    _list_data_sets_path = [   


        [
            "data/toy_example",                         
            "data/toy_example/raw",
            "data/toy_example/splited/train",
            "data/toy_example/splited/val",
            "data/toy_example/db_feature_extractor",
            "data/toy_example/results_consolidated"
        ]


        # [
        #     "data/mnist",             
        #     "../../../../../data_colab/mnist/raw",  ## Apontar o RAW para code_lab
        #     "data/mnist/splited/train",
        #     "data/mnist/splited/val",
        #     "data/mnist/db_feature_extractor",
        #     "data/mnist/results_consolidated"
        # ]



        # ,[
        #     "data/plancton",
        #     "../../../../../data_colab/plancton/raw",  ## Apontar o RAW para code_lab
        #     "data/plancton/splited/train",
        #     "data/plancton/splited/val",
        #     "data/plancton/db_feature_extractor",
        #     "data/plancton/results_consolidated"
        # ]                                   


    ]
