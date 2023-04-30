import os
import glob
import config
config = config.config




_list_paths = config._list_data_sets_path



def run_paths (_list_paths, input_index=True,  input_images=False):

print("[INFO] Deleting Files..:")
for db_paths in _list_paths:	
	for db_path in db_paths:
		# get a list of all files in the folder
		file_list = glob.glob(os.path.join(db_path, "*"))
		# delete each file in the list
		for file_path in file_list:
		    os.remove(file_path)		
			print(db_sub)