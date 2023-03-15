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



# # load all the image paths and randomly shuffle them
# print("[INFO] loading image paths...")
# imagePaths = list(paths.list_images(config.PLANCTON_DATASET_PATH))
# np.random.shuffle(imagePaths)


# ## In case we want to generate split & validation
# # generate training and validation paths
# valPathsLen = int(len(imagePaths) * config.VAL_SPLIT)
# trainPathsLen = len(imagePaths) - valPathsLen
# trainPaths = imagePaths[:trainPathsLen]
# valPaths = imagePaths[trainPathsLen:]

# # copy the training and validation images to their respective
# # directories
# print("[INFO] copying training and validation images...")
# copy_images(trainPaths, config.PLANCTON_TRAIN)
# copy_images(valPaths, config.PLANCTON_VAL)



for db_paths in config._list_data_sets_path:	

	# load all the image paths and randomly shuffle them
	print("[INFO] loading image paths...")
	print("db  ==  " , db_paths[0])
	imagePaths = list(paths.list_images(db_paths[1]))
	np.random.shuffle(imagePaths)


	## In case we want to generate split & validation
	# generate training and validation paths
	valPathsLen = int(len(imagePaths) * config.VAL_SPLIT)
	trainPathsLen = len(imagePaths) - valPathsLen
	trainPaths = imagePaths[:trainPathsLen]
	valPaths = imagePaths[trainPathsLen:]

	# copy the training and validation images to their respective
	# directories
	print("[INFO] copying training and validation images...")
	copy_images(trainPaths, db_paths[2])
	copy_images(valPaths, db_paths[3])
	print("____________________________")