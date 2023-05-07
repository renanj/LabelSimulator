import os
import glob
import sys
import time
import datetime

from aux_functions import f_time_now, f_saved_strings, f_log, f_create_accuracy_chart, f_create_visualization_chart_animation, f_get_files_to_delete, f_delete_files, f_get_subfolders
import config
config = config.config



#Inputs:
_scripts_order = config._scripts_order

_list_data_sets_path = config._list_data_sets_path
_list_train_val = config._list_train_val

_script_name = sys.argv[1]


with open('logs/' + f_time_now(_type='datetime_') + "_delete_files_py" + ".txt", "a") as _f:

	if _script_name in _scripts_order:

		_string_log_input = [0, '[INFO] Starting Simulation Framework']    
	    f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)

	    _string_log_input = [0, '[INFO] num_cores = ' + str(multiprocessing.cpu_count())]    
	    f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)


	    for db_paths in _list_data_sets_path:
	                
	        _string_log_input = [1, '[IMAGE DATABASE] = ' + db_paths[0]]    
	        f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)


	        _string_log_input = [1, '[INFO] Deleting All Files...']
	        f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)        
	        _sub_folders_to_check = f_get_subfolders(db_paths[0])
	        for _sub_folder in _sub_folders_to_check:    
	            f_delete_files(f_get_files_to_delete(_script_name), _sub_folder)   
    else:
		_string_log_input = [0, '[INFO] There is no file with this name ' + _script_name]    
	    f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)
		