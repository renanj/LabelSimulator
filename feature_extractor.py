from xml.dom import ValidationErr
import torch
from torch import optim, nn
from torchvision import models, transforms

import os
from cv2 import cv2
import numpy as np
import pickle as pkl

import config as config
config = config.config

from imutils import paths
import pandas as pd




_models = [
  models.vgg16(pretrained=True),
  models.vgg19(pretrained=True)
  #models.resnet50(pretrained=True)
]

_models_name = [
  'vgg_16',
  'vgg_19'
  #'resnet50'  
]




class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
            # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
            # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
            # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
            # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]
  
    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out) 
        return out 



for db_paths in config._list_data_sets_path:	
    model_name_i = 0
    # _l_train_val = ['train', 'val']
    # _l_train_val = ['train']
    for model in _models:        
        for i in range(len(config._list_train_val)):
            # for train_validation in range(2) --> se fosse fazer para train e validation. Mas no nosso caso so estamos fazendo para train 

            # Initialize the model
            new_model = FeatureExtractor(model)

            # Change the device to GPU
            device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
            new_model = new_model.to(device)


            # Transform the image, so it becomes readable with the model
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(512),
                transforms.Resize(448),
                transforms.ToTensor()                              
                ])


            # Will contain the feature
            features = []
            list_image_names = [] 
            list_image_true_label = [] 
            list_image_manual_label = [] 
            list_image_id = [] 

            
            # TRAIN ou VAL:
            imagePaths = list(paths.list_images(db_paths[i+2])) # 2 and 3


        #     imagePaths = imagePaths[0:5]
            _id_count = 1
            for path in (imagePaths):
                try:
                    img = cv2.imread(path)
                    img = transform(img)
                    img = img.reshape(1, 3, 448, 448)
                    img = img.to(device)  

                    image_name = path.split(os.path.sep)[-1]
                    image_true_label = path.split(os.path.sep)[-2]
                    image_manual_label = "-"
                    image_id = _id_count
                    _id_count = _id_count + 1
                    
                    list_image_names.append(image_name)
                    list_image_true_label.append(image_true_label)
                    list_image_manual_label.append(image_manual_label)
                    list_image_id.append(image_id)
                    
                    
                    with torch.no_grad():
                        # Extract the feature from the image
                        feature = new_model(img)
                        # Convert to NumPy Array, Reshape it, and save it to features variable
                        features.append(feature.cpu().detach().numpy().reshape(-1))                
                    # Convert to NumPy Array                
                except:
                    None
            features = np.array(features) 

            #Dataframe Build:
            _list_of_X = ['X%d' % i for i in range(1, len(features[0])+1, 1)]
            df_X = pd.DataFrame(features, columns=_list_of_X)

            df_names = pd.DataFrame(
                    data=zip(list_image_id,list_image_names,list_image_true_label,list_image_manual_label), 
                    columns=['sample_id','name','labels','manual_label']
                )

            df = pd.concat([df_names,df_X], axis=1)

            
            # checar se diretorio "db_feature"extractor" existe; caso contrario, criar
            if not os.path.exists(db_paths[4]):
                os.makedirs(db_paths[4])


            # checar se sub-folder do modelo existe; caso contrario, criar
            _folder_model_name_path = db_paths[4] + '/' + _models_name[model_name_i]
            
            if not os.path.exists(_folder_model_name_path):
                os.makedirs(_folder_model_name_path)


            _pkl_name = 'df_'+ config._list_train_val[i] + '.pkl'
            _pkl_folder_model_name_path = _folder_model_name_path + '/' + _pkl_name
            df.to_pickle(_pkl_folder_model_name_path) 
            model_name_i = model_name_i + 1

