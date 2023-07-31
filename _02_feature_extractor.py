from xml.dom import ValidationErr
import torch
from torch import optim, nn
from torchvision import models, transforms

import os
# from cv2 import cv2
import cv2
import numpy as np
import pickle 

# import config as config
# config = config.config

# import config
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('test_number')
# args = parser.parse_args()
# config = config.Config(args.test_number)

from config import config

from imutils import paths
import pandas as pd

from tqdm import tqdm_notebook as tqdm

from aux_functions import f_time_now, f_saved_strings, f_log, f_get_files_to_delete, f_delete_files, f_get_subfolders

from sklearn.preprocessing import LabelEncoder


import torch
from torchvision import transforms
from PIL import Image

#Inputs:
_script_name = os.path.basename(__file__)
_GPU_flag = config._GPU_Flag_dict[_script_name]

_list_data_sets_path = config._list_data_sets_path
_list_train_val = config._list_train_val


_models = [
    models.vgg16(pretrained=True),
    models.vgg19(pretrained=True),
    models.resnet50(pretrained=True),
    models.densenet121(pretrained=True),
    models.inception_v3(pretrained=True),
]

_models_name = ['vgg_16', 'vgg_19', 'resnet50', 'densenet121', 'inception_v3']
_models_layer_to_extract = ['classifier.5', 'classifier.3', 'layer4', 'features.norm5', 'Mixed_7c.branch_pool']


_models_transform = []
_models_transform.append(transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
_models_transform.append(transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
_models_transform.append(transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
_models_transform.append(transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
_models_transform.append(transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))




############################################################################################################################################
def extract_features(image_path, model, layer_names=None, transform=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    features = {}

    def hook(layer_name, module_type):
        def hook_fn(module, input, output):
            feature_map = output.detach().cpu().numpy()
            flattened_feature_map = feature_map.reshape(feature_map.shape[0], -1)
            features[layer_name] = {
                'top_layer_name': layer_name.split('.')[0],
                'full_layer_name': layer_name,
                'layer_type': module_type,
                'feature_map': feature_map,
                'flattened_feature_map': flattened_feature_map,
                'shape': flattened_feature_map.shape,
                'dtype': flattened_feature_map.dtype,
                'original_shape': feature_map.shape,
            }
        return hook_fn

    handles = []
    for name, layer in model.named_modules():
        if layer_names is None or name in layer_names:
            handle = layer.register_forward_hook(hook(name, layer.__class__.__name__))
            handles.append(handle)

    _ = model(image)

    for handle in handles:
        handle.remove()

    return features







with open('logs/' + f_time_now(_type='datetime_') + "_02_feature_extractor_py_" + ".txt", "a") as _f:



	print ('[INFO] - Feature Extractor/n/n')
	for db_paths in config._list_data_sets_path:
		print('Path = ', db_paths[0])


		_string_log_input = [1, '[INFO] Deleting All Files...']
		f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)		

		_sub_folders_to_check = f_get_subfolders(db_paths[0])

		for _sub_folder in _sub_folders_to_check:	
			f_delete_files(f_get_files_to_delete(_script_name), _sub_folder)		
			


	

		model_name_i = 0
		# _l_train_val = ['train', 'val']
		# _l_train_val = ['train']
		for model in _models:   
			print('Model = ', model)			 
			for i in range(len(config._list_train_val)):
				print('Cohort = ', config._list_train_val[i])

				features = []
				list_image_names = [] 
				list_image_true_label = [] 
				list_image_manual_label = [] 
				list_image_id = [] 


				df = pd.read_pickle(db_paths[i+2] + '/' + 'df_index_paths_' + config._list_train_val[i] + '.pkl')
				imagePaths = list(df['image_path'].to_list())
				_id_count = 1

				for path in tqdm((imagePaths), colour="green"):
					try:
						feature = extract_features(
												image_path = path, 
												model = _models[model_name_i],
												layer_names = _models_layer_to_extract[model_name_i],
												transform = _models_transform[model_name_i]
									)
						feature = feature[_models_layer_to_extract[model_name_i]]['flattened_feature_map'][0]						
						features.append(feature)


						image_name = path.split(os.path.sep)[-1]
						image_true_label = path.split(os.path.sep)[-2]
						image_manual_label = "-"
						image_id = _id_count
						_id_count = _id_count + 1
						
						list_image_names.append(image_name)
						list_image_true_label.append(image_true_label)
						list_image_manual_label.append(image_manual_label)
						list_image_id.append(image_id)

					except:
						None


				#Dataframe Build:
				_list_of_X = ['X%d' % i for i in range(1, len(features[0])+1, 1)]
				df_X = pd.DataFrame(features, columns=_list_of_X)

				df_names = pd.DataFrame(
						data=zip(list_image_id,
											list_image_names,
											list_image_true_label,
											list_image_manual_label
											), 
						columns=['sample_id','name','labels','manual_label']
					)

				df = pd.concat([df_names,df_X], axis=1)


				# checar se diretorio "db_feature"extractor" existe; caso contrario, criar
				if not os.path.exists(db_paths[4]):
					os.makedirs(db_paths[4])

				print('\n\n\n\n\n\n\n', model_name_i , ' === ', db_paths[4], '/', _models_name[model_name_i], '\n\n\n\n\n\n\n\n')

				# checar se sub-folder do modelo existe; caso contrario, criar
				_folder_model_name_path = db_paths[4] + '/' + _models_name[model_name_i]

				if not os.path.exists(_folder_model_name_path):
					os.makedirs(_folder_model_name_path)


				_pkl_name = 'df_'+ config._list_train_val[i] + '.pkl'
				_pkl_folder_model_name_path = _folder_model_name_path + '/' + _pkl_name
				df.to_pickle(_pkl_folder_model_name_path) 
			model_name_i = model_name_i + 1




		##### Export Encoder.pkl  --- (for all possible labels / y-variables):

		model_name_i = 0
		_list_df_all_labels = []
		for model in _models:   
			print('Model = ', model)			 
			for i in range(len(config._list_train_val)):				

				_df_temp = pd.read_pickle(db_paths[4] + '/' + _models_name[model_name_i] + '/' + 'df_'+ config._list_train_val[i] + '.pkl')				
				_list_df_all_labels.append(_df_temp)
			
			df_all_labels = pd.concat(_list_df_all_labels, axis=0)
			df_all_labels = df_all_labels.reset_index(drop=True)
			all_labels = df_all_labels['labels'].values
			label_encoder = LabelEncoder()
			label_encoder.fit(all_labels)

			pickle.dump(label_encoder, open(db_paths[4] + '/' + _models_name[model_name_i] + '/' + 'label_encoder.pkl', 'wb'))
			# pickle.load(open('/content/drive/MyDrive/Mestrado/Git/LabelSimulator/data/plancton/db_feature_extractor/vgg_16/label_encoder.pkl', 'rb')) 
			model_name_i = model_name_i + 1

			print('---------------/n/n')

