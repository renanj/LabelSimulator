import subprocess


def label_simulator_run():

    scripts_dict = {        
        1: '_01_build_dataset.py',
        2: '_02_feature_extractor.py',
        3: '_03_dim_reduction.py',
        4: '_04_generator_faiss.py',
        5: '_05_framework.py',
        6: '_06_generate_visualization.py'        
    }
    
    print("Hi - this is LabelSimulator Executor\n")
    print("Please, check below the Scripts and related Numbers:")
    for _key in scripts_dict.keys():
        print(_key, " = ", scripts_dict[_key])

    print('9: ALL Scripts')    
    input_list = input("Enter the code you want to run splited by spaces: ").split()
    test_number = input("Enter the Test Name and Number: ")
    os.environ['TEST_NUMBER'] = test_number  # set the environment variable


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
                print(scripts_dict[_key])
                subprocess.run(['python3', scripts_dict[_key], test_number])            
                
        else:
            for i_script in range(len(input_list)):
                print(scripts_dict[int(input_list[i_script])])
                subprocess.run(['python3', scripts_dict[int(input_list[i_script])], test_number])
                


label_simulator_run()
