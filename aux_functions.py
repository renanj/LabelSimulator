import time 
import datetime
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from sklearn.datasets import make_blobs

import os
import glob



import config

# Get the test_number from the environment variable
test_number = os.getenv('TEST_NUMBER')
if test_number is None:
    raise ValueError('TEST_NUMBER environment variable not set. Please provide a test number.')

config = config.Config(test_number)

#Inputs
_scripts_order = config._scripts_order
_files_generated = config._files_generated

def create_label_encoder_obj(root_dir):

  classes = sorted(os.listdir(root_dir))
  classes = [class_name for class_name in classes if os.path.isdir(os.path.join(root_dir, class_name)) and class_name != ".ipynb_checkpoints"]  

  
  #LabelEncoder() Object
  lb = LabelEncoder()  
  classes_adjusted = classes.copy() + ['-']  
  lb.fit(classes_adjusted)  

  return lb



class CustomImageDataset():

    def __init__(self, root_dir, images_path_to_use, label_encoder=None, max_size=None, transform=None, target_transform=None, validation_transform=None, validation_target_transform=None):
    
        # validacao max_size tem que ser no minimo o numero minumo de classes ( a validacao tem que ser pelo validation... pq tem menos dados.. ou pelo menos.. )
        # criar get item pelo sample_id 

        self.root_dir = root_dir        
        self.max_size = max_size
        self.transform = transform
        self.target_transform = target_transform
        self.label_encoder = label_encoder
        self.image_sample_id, self.image_name, self.image_paths, self.label_true_original, self.label_true_encoded, self.label_manual_original, self.label_manual_encoded = self.get_image_info()
    
        self.shuffle_dataset()

    

    def shuffle_dataset(self):
        # Shuffle the indices of the samples
        indices = list(range(len(self.image_paths)))
        random.shuffle(indices)

        # Use the shuffled indices to re-order  the lists
        self.image_sample_id = [self.image_sample_id[i] for i in indices]
        self.image_name = [self.image_name[i] for i in indices]
        self.image_paths = [self.image_paths[i] for i in indices]
        self.label_true_original = [self.label_true_original[i] for i in indices]
        self.label_true_encoded = [self.label_true_encoded[i] for i in indices]
        self.label_manual_original = [self.label_manual_original[i] for i in indices]
        self.label_manual_encoded = [self.label_manual_encoded[i] for i in indices]


    def get_image_info(self):
        
        
        image_sample_id, image_name, image_paths = [], [], []
        label_true_original, label_true_encoded = [], []
        label_manual_original, label_manual_encoded = [], []
        
        classes = sorted(os.listdir(self.root_dir))
        classes = [class_name for class_name in classes if os.path.isdir(os.path.join(self.root_dir, class_name)) and class_name != ".ipynb_checkpoints"]

        #MaxSize Buckets Per Class:
        if self.max_size is not None:            
            if self.max_size < len(classes):
                self.max_size = len(classes)

            max_size_buckets = self.max_size // len(classes)
        else:
          max_size_buckets = None  
    

        _sample_id_count = 1
        for class_name in classes:

            class_path = os.path.join(self.root_dir, class_name)          
            if os.path.isdir(class_path):                
                try:                   
                    image_files = os.listdir(class_path)                      
                    temp_list = [class_path + '/' + filename for filename in image_files]

                    set_1 = set(images_path_to_use)
                    set_2 = set(temp_list)
                    image_files = set_1.intersection(set_2)
                    image_files = list(image_files)
                    image_files = [item.split('/')[-1] for item in image_files]

                                
                    #Limitar o numero de samples per class de acordo com o "max_size"
                    if self.max_size is not None:                        
                        image_files = image_files[:max_size_buckets]                        


                    image_sample_id.extend([num for num in range(_sample_id_count, _sample_id_count+len(image_files))])                
                    _sample_id_count = _sample_id_count + len(image_files)

                    image_name.extend(image_files)
                    image_paths.extend([os.path.join(class_path, file) for file in image_files])

                    label_true_original.extend([class_name for file in image_files])
                    label_manual_original.extend(['-' for file in image_files])

                    if label_encoder:
                      label_true_encoded.extend([lb.transform([class_name])[0] for file in image_files])
                      label_manual_encoded.extend([lb.transform(['-'])[0] for file in image_files])
                
                except:
                    None

        # return image_paths, name, sample_id, labels, manual_label
        return  image_sample_id, image_name, image_paths, label_true_original, label_true_encoded, label_manual_original, label_manual_encoded



    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx, flag=False):

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        class_label = self.label_true_encoded[idx]
        if self.target_transform:
            class_label = self.target_transform(class_label)

        image_name = self.image_name[idx]
        image_sample_id = self.image_sample_id[idx]
        image_paths = self.image_paths[idx]
        label_true_original = self.label_true_original[idx]


        if flag == False:
            return image, class_label
        else:
            return image, class_label, image_name, image_sample_id, image_paths, label_true_original



    def get_item_by_sample_id(self, sample_id):
        # Find the index of the given sample_id in the list of image_sample_id
        try:
            idx = self.image_sample_id.index(sample_id)
        except ValueError:
            raise ValueError(f"Sample ID {sample_id} not found in the dataset.")

        # Return the associated values for the given sample_id
        image_name = self.image_name[idx]
        image_path = self.image_paths[idx]
        class_label = self.label_true_encoded[idx]
        label_true_original = self.label_true_original[idx]

        return image_name, image_path, class_label, label_true_original






def is_list_of_lists(obj):
    return isinstance(obj, list) and all(isinstance(sublist, list) for sublist in obj)


def is_list_of_lists_of_lists(variable):
    if not isinstance(variable, list):
        return False

    for item in variable:
        if not isinstance(item, list):
            return False
        for subitem in item:
            if not isinstance(subitem, list):
                return False

    return True


def transform_to_list_of_lists(variable):
    if not isinstance(variable, list):
        return variable

    if all(isinstance(item, list) for item in variable):
        return [transform_to_list_of_lists(item[0]) for item in variable]

    return variable



def to_list_of_lists(obj):
    if not isinstance(obj, list):
        obj = [obj]

    return [obj] 


def f_get_subfolders(path):

	sub_folders = []
	for entry in os.scandir(path):
		if entry.is_dir() and not entry.name.startswith('.'):
			sub_folders.append(entry.path)
			sub_folders += f_get_subfolders(entry.path)
	return sub_folders


def f_get_files_to_delete(script_name, _scripts_order=_scripts_order, _files_generated=_files_generated):	

	position = _scripts_order.index(script_name)

	_temp_scripts_to_check = []
	_files_to_delete = []
	for i in range(position, len(_scripts_order)):  
		_temp_scripts_to_check.append(_scripts_order[i])  
	for _k in _files_generated.keys():
		if _k in _temp_scripts_to_check:
			_files_to_delete = _files_to_delete + _files_generated[_k]   
	return _files_to_delete






def f_delete_files (list_files_to_delete, _path, NOT_DELETE_files_list=[None]):		
	if 'raw' in _path:				
		None		
	else:					   
		file_list = os.listdir(_path)				
		for file in file_list:
			if file in list_files_to_delete:
				if file in NOT_DELETE_files_list:
					None
				else:
					try:
						os.remove(_path + '/' + file)	 
						print("Delelted File = ", file)
					except:
						None
			else:
				None




def f_time_now(_type='datetime'):
	if _type == 'datetime':
		return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
	if _type == 'datetime_':
		return datetime.datetime.utcnow().strftime("%Y-%m-%d_%H:%M")		
	elif _type == 'date':
		return datetime.datetime.utcnow().strftime("%Y-%m-%d")
	elif _type == 'hour':
		return datetime.datetime.utcnow().strftime("%H:%M:%S")
	else:
		return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")


def f_saved_strings(_string):

	if _string == 'line_split_01':
		_string = '- ' * 10
		return _string
	else:
		return _string


def f_log(_string, _level, _file):


	_levels_allowed = [0,1,2,3,4,5,6,7,8,9,10,11,12]
	_string = f_saved_strings(_string)

	try:
		if _level not in _levels_allowed:
			_file.write('')
			_file.flush				
			return None 
		else:					
			#write
			_write_string = ('\t' * _level) + _string + '\n'
			_file.write(_write_string)
			_file.flush()

			#print
			_print_string = ('  ' * _level) + _string
			return print(_print_string)
	except:		
		return None



def generate_data(n_samples, cluster_std_ratio, n_outliers):
	# Generate data with 2 clusters where cluster 1 is more disperse than cluster 2
	X, y = make_blobs(n_samples=n_samples-n_outliers, centers=2, cluster_std=[1.5*cluster_std_ratio, 0.5], random_state=42)
	outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, 2))
	X = np.concatenate((X, outliers), axis=0)
	y = np.concatenate((y, np.full((n_outliers,), fill_value=-1)), axis=0)

	# Create a dataframe with X1 and X2
	df = pd.DataFrame(X, columns=['X1', 'X2'])	
	df['name'] = ['name_' + str(i) for i in  range(1, n_samples+1)]
	df['labels'] = y
	df['manual_label'] = "-"
	df['sample_id'] = range(1, n_samples+1)  # Add sample IDs starting from 1	
	
	df = df[['sample_id','name','labels','manual_label','X1','X2']]
	return df


def closest_value(row):
	non_null_values = row.dropna()
	if non_null_values.empty:
		return None
	else:
		return non_null_values.iloc[0]
		


def f_create_visualization_chart_animation(_df_2D, _path, _file_name, _list_simulation_names, _list_selected_samples, _n_fractions, _fps=3, _only_2D_chart=False):


	plt.figure() 

	if is_list_of_lists(_list_selected_samples) == False:
		_list_selected_samples = to_list_of_lists(_list_selected_samples)

	if is_list_of_lists_of_lists(_list_selected_samples) == True:
		_list_selected_samples = transform_to_list_of_lists(_list_selected_samples)




	_temp_X_columns = list(_df_2D.loc[:,_df_2D.columns.str.startswith("X")].columns)
	if _temp_X_columns == None:
	  _temp_X_columns = list(_df_2D.loc[:,_df_2D.columns.str.startswith("X")].columns)	


	if len(_temp_X_columns) > 2:
		print("This Arcthrecute has more than 2 dimensions... Path = ", _path)
		return None

	_df_2D = _df_2D[['sample_id', 'X1', 'X2']]


	num_simulations = len(_list_selected_samples)
	nrows = 1
	ncols = num_simulations
	figsize = (4 * num_simulations, 4)

	fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, tight_layout=True)

	if num_simulations == 1:
		axs = [axs]


	scatter_plots = []
	for i in range(num_simulations):
		sample_ids = _list_selected_samples[i]
		df_subset = _df_2D.copy()
		df_subset['color'] = 'bisque'
		df_subset['fraction'] = ''
		scatter_plot = axs[i].scatter(df_subset['X1'], df_subset['X2'], c=df_subset['color'])
		scatter_plots.append(scatter_plot)
		axs[i].set_title(_list_simulation_names[i])
	

	def animate(i):
		for j in range(num_simulations):
			sample_ids = _list_selected_samples[j]
			df_subset = _df_2D.copy()
			df_subset['color'] = 'bisque'
			index = i % len(sample_ids)
			df_subset.loc[df_subset['sample_id'].isin(sample_ids[:index+1]), 'color'] = 'slategray'
			df_subset.loc[df_subset.index <= math.ceil((index+1)/len(sample_ids)*_n_fractions)*len(df_subset)/_n_fractions,'fraction'] = f"{math.ceil((index+1)/len(sample_ids)*100)}%"
			scatter_plots[j].set_color(df_subset['color'])			
			axs[j].set_xlabel('fraction')


		# axs[0].table(cellText=[df_subset['fraction'].unique()], loc='bottom', cellLoc='center')
		# axs[0].axis('off')

	if _only_2D_chart == False:
		ani = animation.FuncAnimation(fig, animate, frames=len(_list_selected_samples[0]), interval=100, repeat=True)
		ani.save(f'{_path}/{_file_name}.gif', writer='imagemagick', fps=_fps)	
		
	_df_2D['color'] = 'bisque'
	
	scatter_plot = axs[0].scatter(df_subset['X1'], df_subset['X2'], c='bisque')
	fig2, axs2 = plt.subplots(nrows=_n_fractions, ncols=num_simulations, figsize=(4*num_simulations, 4*_n_fractions), tight_layout=True)	

	i = 0
	for i in range(_n_fractions):
	  for j in range(num_simulations):
	    sample_ids = _list_selected_samples[j]
	    df_subset = _df_2D.copy()
	    df_subset['color'] = 'bisque'
	    df_subset['fraction'] = ''
	    index = math.ceil((i+1)/_n_fractions*len(sample_ids))
	    df_subset.loc[df_subset['sample_id'].isin(sample_ids[:index]), 'color'] = 'slategray'
	    if num_simulations > 1:
	      axs2[i][j].scatter(df_subset['X1'], df_subset['X2'], c=df_subset['color'])
	      axs2[i][j].set_title(_list_simulation_names[j])
	      axs2[i][j].set_xlabel('X1')
	      axs2[i][j].set_ylabel('X2')
	    else:
	      axs2[i].scatter(df_subset['X1'], df_subset['X2'], c=df_subset['color'])
	      axs2[i].set_title(_list_simulation_names[j])
	      axs2[i].set_xlabel('X1')
	      axs2[i].set_ylabel('X2')

	  if num_simulations > 1:      
	    axs2[i][0].set_ylabel(f"{math.ceil((i+1)/_n_fractions*100)}% of samples selected")	
	  else:
	    axs2[i].set_ylabel(f"{math.ceil((i+1)/_n_fractions*100)}% of samples selected")			
		
	fig2.subplots_adjust(top=0.85)		
	fig2.savefig(f'{_path}/{_file_name}.png')	




def f_create_consolidate_accuracy_chart(_df, _path, _file_name, _col_x, _col_y, _hue):

	_list_cols = [_col_x, _col_y, _hue]
	
	_temp_df_chart = _df[_list_cols]	
	_temp_df_chart = _temp_df_chart.reset_index(drop=True)					

	
	#[TO-DO] change this to be dynamic... 
	# palette = sns.color_palette("mako", len(_list_models))						
	figure = plt.figure()
	ax = figure.add_subplot(1, 1, 1)

	# palette = ['#F22B00', '#40498e', '#357ba3', '#38aaac', '#79d6ae']	


	palette_hue_colors = {
		'Random': '#000000',
		'Equal_Spread': '#1f78b4',
		'Dense_Areas_First': '#b2df8a',
		'Centroids_First': '#33a02c',
		'Outliers_First': '#fb9a99',
		'Uncertainty': '#e31a1c',
		'Entropy': '#fdbf6f',
		'Margin': '#ff7f00',
		'Bald': '#cab2d6',
		'BatchBALD': '#521570',		
		'Equal_Spread_2D' : '#6a3d9a', 
		'Dense_Areas_First_2D' : '#ffff99',
		'Centroids_First_2D' : '#b15928', 
		'Outliers_First_2D' : '#a6cee3'
	}

	sns.set(rc={'figure.figsize':(15.7,8.27)})	
	sns.set_style('white')
	_chart = sns.lineplot(data=_temp_df_chart, 
				x=_col_x, 
				y=_col_y, 
				hue=_hue,
				)

	figure = _chart.get_figure()
	figure.savefig(_path + '/' + _file_name)
	figure.savefig(f'{_path}/{_file_name}.png')


def f_create_random_vs_query_accuracy_chart(_df, _path, _file_name,  _col_x, _col_y, _hue):

	# Filter out the "Random" query strategy from the dataframe
    filtered_df = _df[_df['Query_Strategy'] != 'Random']

    # Get the unique query strategies
    query_strategies = filtered_df['Query_Strategy'].unique()

    # Define the color palette based on the provided dictionary
    palette_hue_colors = {
		'Random': '#000000',
		'Equal_Spread': '#1f78b4',
		'Dense_Areas_First': '#b2df8a',
		'Centroids_First': '#33a02c',
		'Outliers_First': '#fb9a99',
		'Uncertainty': '#e31a1c',
		'Entropy': '#fdbf6f',
		'Margin': '#ff7f00',
		'Bald': '#cab2d6',
		'BatchBALD': '#521570',		
		'Equal_Spread_2D' : '#6a3d9a', 
		'Dense_Areas_First_2D' : '#ffff99',
		'Centroids_First_2D' : '#b15928', 
		'Outliers_First_2D' : '#a6cee3'
	}


    # Calculate the number of rows and columns for the subplots
    num_query_strategies = len(query_strategies)
    num_cols = math.ceil(math.sqrt(num_query_strategies))
    num_rows = math.ceil(num_query_strategies / num_cols)

    # Create the figure and subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 10))

    # Flatten the axs array to make it easier to iterate
    axs = axs.ravel()

    # Iterate over each query strategy and plot the line chart in each subplot
    for i, query_strategy in enumerate(query_strategies):
        ax = axs[i] if num_query_strategies > 1 else axs  # Select the current subplot

        # Retrieve the data for the current query strategy and "Random"
        query_strategy_data = filtered_df[filtered_df['Query_Strategy'] == query_strategy]
        random_data = _df[_df['Query_Strategy'] == 'Random']

        # Plot the lines
        _chart = sns.lineplot(x=_col_x, y=_col_y, hue=_hue, palette=palette_hue_colors,
                     data=pd.concat([query_strategy_data, random_data]), ax=ax)

        # ax.set_ylabel('Samples Accuracy Validation')
        # ax.set_xlabel('Interaction')					
        ax.set_title(f'Comparison: Random vs. {query_strategy}')
        ax.legend()

    # Remove any unused subplots
    for j in range(num_query_strategies, num_rows * num_cols):
        fig.delaxes(axs[j])

    # Adjust the spacing between subplots
    fig.tight_layout()    	
    # fig.savefig(_path)
    fig.savefig(f'{_path}/{_file_name}.png')
    


# ================================================================================ 


import os
import tensorflow as tf
from PIL import Image

def extract_mnist(output_folder):

	def download_mnist():
		# Download the MNIST dataset from TensorFlow
		(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
		return x_train, y_train

	def create_png_images(mnist_data, mnist_labels, output_folder):
		os.makedirs(output_folder, exist_ok=True)

		for i, (image, label) in enumerate(zip(mnist_data, mnist_labels)):
			label_folder = os.path.join(output_folder, str(label))
			os.makedirs(label_folder, exist_ok=True)

			image_path = os.path.join(label_folder, f"{i}.png")
			image = Image.fromarray(image)  # Convert NumPy array to PIL image
			image.save(image_path)
			print(f"Saved image {i}.png in folder {label}")


	mnist_data, mnist_labels = download_mnist()    
	create_png_images(mnist_data, mnist_labels, output_folder)




