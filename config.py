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
    BATCH_SIZE = 8
    VAL_SPLIT = 0



    # 0: RAW
    # 1: SPLITED-TRAIN
    # 2: SPLITED-TEST        
    # 3: DB

    _list_simulation_sample_name = ['Random', 'NSS', 'SPB','DEN', 'OUT']
    _list_train_val = ['train'] #, 'val'
    #_list_train_val = ['val'] #, 'val'
    #controller: 
    _list_data_sets_path = [      

        [
            "data/mnist",             
            "../../../../../q/mnist/raw",  ## Apontar o RAW para code_lab
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
