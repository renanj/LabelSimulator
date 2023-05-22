# import the necessary packages
from torchvision.datasets import ImageFolder, MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import shutil
import os
import config
config = config.config

#Inputs:
_script_name = os.path.basename(__file__)
_GPU_flag = config._GPU_Flag_dict[_script_name]

_list_data_sets_path = config._list_data_sets_path
_list_train_val = config._list_train_val


import pandas as pd

from aux_functions import f_time_now, f_saved_strings, f_log, f_create_accuracy_chart, f_create_visualization_chart_animation, f_get_files_to_delete, f_delete_files, f_get_subfolders




def copy_images(imagePaths, folder):
	# check if the destination folder exists and if not create it
	if not os.path.exists(folder):
		os.makedirs(folder)


	# loop over the image pathsPLANCTON_DATASET_PATH
	for path in imagePaths:
		# grab image name and its label from the path and create
		# a placeholder corresponding to the separate label folder
		imageName = path.split(os.path.sep)[-1]
		label = path.split(os.path.sep)[-2]
		labelFolder = os.path.join(folder, label)

		# check to see if the label folder exists and if not create it
		if not os.path.exists(labelFolder):
			os.makedirs(labelFolder)

		# construct the destination image path and copy the current
		# image to it
		destination = os.path.join(labelFolder, imageName)
		shutil.copy(path, destination)



# def run_paths (_list_paths, input_index=True,  input_images=False):

_list_paths = config._list_data_sets_path
input_index=True  
input_images=False

with open('logs/' + f_time_now(_type='datetime_') + "_06_framework_py_" + ".txt", "a") as _f:

	for db_paths in _list_paths:	

		_string_log_input = [1, '[INFO] Deleting All Files...']
		f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)		
		_sub_folders_to_check = f_get_subfolders(db_paths[0])
		for _sub_folder in _sub_folders_to_check:	
			f_delete_files(f_get_files_to_delete(_script_name), _sub_folder)				

		#Create directory for everyone that is NOT RAW
		for db_sub in db_paths:
			print(db_sub)
			if 'raw' in db_sub:
				None
			else:
				if not os.path.exists(db_sub):
					os.makedirs(db_sub)	
					file_path = os.path.join(db_sub, '.gitkeep')
					with open(file_path, 'w') as f:
						f.write('')					


		# load all the image paths and randomly shuffle them
		print("[INFO] loading image paths...")
		print("db  ==  " , db_paths[0])
		imagePaths = list(paths.list_images(db_paths[1]))
		np.random.shuffle(imagePaths)


		## In case we want to generate split & validation
		# generate training and validation paths
		valPathsLen = int(len(imagePaths) * config.VAL_SPLIT)
		trainPathsLen = len(imagePaths) - valPathsLen

		
		#delete
		valPaths = imagePaths[50:100]
		trainPaths = imagePaths[:50]				
		#delete ------- 
		
		# valPaths = imagePaths[trainPathsLen:]
		# trainPaths = imagePaths[:trainPathsLen]		

		# if valPathsLen > 2000:			
		# 	valPaths = imagePaths[trainPathsLen:(trainPathsLen*4)]			


		#Create pkl.index... with train and val
		df = pd.DataFrame(trainPaths, columns=['image_path'])
		df.to_pickle(db_paths[2]  + '/' + 'df_index_paths_train.pkl')			 


		df = pd.DataFrame(valPaths, columns=['image_path'])
		df.to_pickle(db_paths[3]  + '/' + 'df_index_paths_validation.pkl')			 


		if input_images == True: 
			# copy the training and validation images to their respective
			# directories
			print("[INFO] copying training and validation images...")
			copy_images(trainPaths, db_paths[2])
			copy_images(valPaths, db_paths[3])
			# print("valPath = ", valPaths)

			# print("db_paths[2] = ", db_paths[2])

			# print("trainPaths = ", trainPaths)
			# print("db_paths[3] = ", db_paths[3])
			print("____________________________")
		else:
			print("[INFO] images will NOT be copied to train and validation paths")
			None



# run_paths(
#	 _list_paths = config._list_data_sets_path,
#	 input_index=True,  
#	 input_images=False
# )
