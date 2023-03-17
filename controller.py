import subprocess


def label_simulator_run():

    scripts_dict = {        
        1: 'build_dataset.py',
        2: 'feature_extractor.py',
        3: 'simulation_framework.py',
        4: 'dim_reduction.py'        
    }
    
    print("Hi - this is LabelSimulator Executor\n")
    print("Please, check below the Scripts and related Numbers:")
    for _key in scripts_dict.keys():
        print(_key, " = ", scripts_dict[_key])

    print('9: ALL Scripts')    
    input_list = input("Enter the code you want to run splited by spaces: ").split()

    _flag_break = 0
    for i_script in range(len(input_list)):        

      if list(scripts_dict.keys()).count(int(input_list[i_script])) == 0:
        if input_list[i_script] == '9':
          None
        else:
            print("ERROR!")
            print("The input", input_list[i_script], " is not valid!")
            _flag_break = _flag_break + 1
            break

    if _flag_break > 0:
        print("Run again please")
    else:
        if input_list.count('9') > 0:
           for _key in scripts_dict.keys():
                #subprocess.run(['python3', scripts_dict[_key]])            
                print(scripts_dict[_key])
        else:
            for i_script in range(len(input_list)):
                #subprocess.run(['python3', scripts_dict[int(input_list[i_script])]])
                print(scripts_dict[int(input_list[i_script])])


label_simulator_run()                