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
    #controller: 
    _list_data_sets_path = [
        # [
        #     "data/plancton",
        #     "data/plancton/raw",  
        #     "data/plancton/splited/train",
        #     "data/plancton/splited/val",
        #     "data/plancton/db_feature_extractor",
        #     "data/plancton/results_consolidated"
        # ],

        # [
        #     "data/mnist",
        #     "data/mnist/raw",  
        #     "data/mnist/splited/train",
        #     "data/mnist/splited/val",
        #     "data/mnist/db_feature_extractor",
        #     "data/mnist/results_consolidated"
        # ]                   


        [
            "../data_driver/mnist",
            "../data_driver/mnist/raw",  
            "../data_driver/mnist/splited/train",
            "../data_driver/mnist/splited/val",
            "../data_driver/mnist/db_feature_extractor",
            "../data_driver/mnist/results_consolidated"
        ],

        [
            "../data_driver/plancton",
            "../data_driver/plancton/raw",  
            "../data_driver/plancton/splited/train",
            "../data_driver/plancton/splited/val",
            "../data_driver/plancton/db_feature_extractor",
            "../data_driver/plancton/results_consolidated"
        ],                                   

        # [
        #     "../data_driver/test_1",
        #     "../data_driver/test_1/raw",  
        #     "../data_driver/test_1/splited/train",
        #     "../data_driver/test_1/splited/val",
        #     "../data_driver/test_1/db_feature_extractor",
        #     "../data_driver/test_1/results_consolidated"
        # ],

        # [
        #     "../data_driver/test_2",
        #     "../data_driver/test_2/raw",  
        #     "../data_driver/test_2/splited/train",
        #     "../data_driver/test_2/splited/val",
        #     "../data_driver/test_2/db_feature_extractor",
        #     "../data_driver/test_2/results_consolidated"
        # ],                                   
    ]




















    

