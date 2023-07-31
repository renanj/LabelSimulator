import argparse
import os
import glob
import sys
import time
import datetime

# Initialize the ArgumentParser object
parser = argparse.ArgumentParser(description='Delete files script')

# Add positional arguments
parser.add_argument('script_name', help='The name of the script')
parser.add_argument('test_number', help='The test number to use')

# Parse the arguments
args = parser.parse_args()

_script_name = args.script_name
test_number = args.test_number

os.environ['TEST_NUMBER'] = test_number  # set the environment variable

print("AQUI!!!", _script_name)
print("ESSSE!!!", test_number)

from aux_functions import f_time_now, f_saved_strings, f_log, f_create_visualization_chart_animation, f_get_files_to_delete, f_delete_files, f_get_subfolders
from config import config

# config = Config()  # Initialize Config class with test_number

#Inputs:
_scripts_order = config._scripts_order
_list_data_sets_path = config._list_data_sets_path
_list_train_val = config._list_train_val

_list_files_to_delete = f_get_files_to_delete(_script_name)
with open('logs/' + f_time_now(_type='datetime_') + "_delete_files_py_" + ".txt", "a") as _f:
    _string_log_input = [0, '[INFO] Deleting Files Script']
    f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)

    if _script_name in _scripts_order:
        for db_paths in _list_data_sets_path:
            _string_log_input = [1, '[IMAGE DATABASE] = ' + db_paths[0]]
            f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)

            _string_log_input = [1, '[INFO] Deleting All Files...']
            f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)
            _sub_folders_to_check = f_get_subfolders(db_paths[0])
            for _sub_folder in _sub_folders_to_check:
                f_delete_files(_list_files_to_delete, _sub_folder)
    else:
        _string_log_input = [0, '[INFO] There is no file with this name ' + _script_name]
        f_log(_string = _string_log_input[1], _level = _string_log_input[0], _file = _f)




