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
config = config.config

#Inputs
_scripts_order = config._scripts_order
_files_generated = config._files_generated


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




def f_delete_files (list_files_to_delete, _path):        
    if 'raw' in _path:                
        None        
    else:                       
        file_list = b m
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



    _string_log_input = ['[INFO] Starting Dimension Reduction', 0]    

    f_print(_string=_string, _level=__level)
    f_write(f_print(_string=_string, _level=_level, _write_option=True), )




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
        




# def f_generate_gif_chart_multiple_simulation(_df,_file_name_png, _file_name_gif,  _list_selected_samples, _list_chart_titles, fractions, _fps=3):
#     num_simulations = len(_list_selected_samples)
#     nrows = 1
#     ncols = num_simulations
#     figsize = (4 * num_simulations, 4)

#     fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, tight_layout=True)

#     if num_simulations == 1:
#         axs = [axs]



#     scatter_plots = []
#     for i in range(num_simulations):
#         sample_ids = _list_selected_samples[i]
#         df_subset = _df.copy()
#         df_subset['color'] = 'whitesmoke'
#         df_subset['fraction'] = ''
#         scatter_plot = axs[i].scatter(df_subset['X1'], df_subset['X2'], c=df_subset['color'])
#         scatter_plots.append(scatter_plot)
#         axs[i].set_title(_list_chart_titles[i])


#     def animate(i):
#         for j in range(num_simulations):
#             sample_ids = _list_selected_samples[j]
#             df_subset = _df.copy()
#             df_subset['color'] = 'whitesmoke'
#             index = i % len(sample_ids)
#             df_subset.loc[df_subset['sample_id'].isin(sample_ids[:index+1]), 'color'] = 'blue'
#             df_subset.loc[df_subset.index <= math.ceil((index+1)/len(sample_ids)*fractions)*len(df_subset)/fractions,'fraction'] = f"{math.ceil((index+1)/len(sample_ids)*100)}%"
#             scatter_plots[j].set_color(df_subset['color'])
#             axs[j].set_xlabel('fraction')

#         # axs[0].table(cellText=[df_subset['fraction'].unique()], loc='bottom', cellLoc='center')
#         # axs[0].axis('off')

#     anim = animation.FuncAnimation(fig, animate, frames=len(_list_selected_samples[0]), interval=100, repeat=True)
#     anim.save(_file_name_gif + '.gif', writer='imagemagick', fps=_fps)

#     df_subset.loc[df_subset['sample_id'].isin(sample_ids[:index+1]), 'color'] = 'blue'
#     _df['color'] = 'whitesmoke'
    
#     scatter_plot = axs[0].scatter(df_subset['X1'], df_subset['X2'], c='whitesmoke')
#     fig2, axs2 = plt.subplots(nrows=fractions, ncols=num_simulations, figsize=(4*num_simulations, 4*fractions), tight_layout=True)
#     i = 0
#     for i in range(fractions):
#         for j in range(num_simulations):
#             sample_ids = _list_selected_samples[j]
#             df_subset = _df.copy()
#             df_subset['color'] = 'whitesmoke'
#             df_subset['fraction'] = ''
#             index = math.ceil((i+1)/fractions*len(sample_ids))
#             df_subset.loc[df_subset['sample_id'].isin(sample_ids[:index]), 'color'] = 'red'
#             axs2[i][j].scatter(df_subset['X1'], df_subset['X2'], c=df_subset['color'])
#             axs2[i][j].set_title(_list_chart_titles[j])
#             axs2[i][j].set_xlabel('X1')
#             axs2[i][j].set_ylabel('X2')
#         axs2[i][0].set_ylabel(f"{math.ceil((i+1)/fractions*100)}% of samples selected")    
        
#     plt.savefig(_file_name_png + '.png')



# def f_generate_gif_chart_one_simulation(_df, _path, _selected_samples, _n_charts, _chart_title, _fps=5):
#     # define the figure size and layout

#     _temp_X_columns = [x for x, mask in zip(_df.columns.values, _df.columns.str.startswith("X")) if mask]
#     if len(_temp_X_columns) > 2:
#         print("Aborting... The dataframe is not valid. It has more than two [X1, X2] dimensions")
#         return None


#     nrows = int(_n_charts ** 0.5)
#     ncols = int(_n_charts / nrows)
#     if _n_charts > nrows * ncols:
#         ncols += 1

#     fig_size = (ncols * 6, nrows * 5)        
        
          
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size)

#     _color_start = 'darkblue'
#     _color_end = 'lightwhitesmoke'

#     # create a scatter plot for each chart
#     for i, ax in enumerate(axes.flat):
#         # calculate the number of selected samples for the current chart
#         n_samples = int(len(_selected_samples) * ((i + 1) / _n_charts))
#         samples = _selected_samples[:n_samples]

#         # create the scatter plot
#         colors = [_color_start if s in samples else _color_end for s in _df['sample_id']]
#         ax.scatter(_df['X1'], _df['X2'], c=colors)
#         ax.set_title(f'{int(((i + 1) / _n_charts) * 100)}% of selected samples')

#     # add the chart title to the top center of the figure
#     fig.suptitle(_chart_title, fontsize=16, y=1.05, x=0.5)

#     # adjust the spacing between subplots
#     fig.tight_layout()


#     # save the figure as a .png file
#     if _path != None:
#         fig.savefig(f'{_path}/{_chart_title}.png', dpi=300)        
#     else:
#         fig.savefig(f'{_chart_title}.png', dpi=300)
        

#     # create a scatter plot with all data points
#     fig, ax = plt.subplots()
#     sc = ax.scatter(_df['X1'], _df['X2'], c='whitesmoke')

#     # define the update function for the animation
#     def update(frame):
#         colors = [_color_start if s in _selected_samples[:frame+1] else _color_end for s in _df['sample_id']]
#         sc.set_color(colors)

#         # add the chart title to the top center of the figure
#         ax.set_title(_chart_title, fontsize=16, y=1.05, x=0.5)

#         return sc,

#     # create the animation object
#     ani = animation.FuncAnimation(fig, update, frames=len(_selected_samples), interval=1000, blit=True)

#     # save the animation as a .gif file    
    


#     # save the figure as a .png file
#     if _path != None:
#         ani.save(f'{_path}/{_chart_title}.gif', writer='imagemagick', fps=_fps)            
#     else:
#         ani.save(f'{_chart_title}.gif', writer='imagemagick', fps=_fps)    
        
        


#     # show the figure
#     # plt.show()
#     return print("Exported Chart .png and .gif")


def f_create_visualization_chart_animation(_df_2D, _path, _file_name, _list_simulation_names, _list_selected_samples, _n_fractions, _fps=3):
    
    # _df_2D  = X1, X2

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
            axs2[i][j].scatter(df_subset['X1'], df_subset['X2'], c=df_subset['color'])
            axs2[i][j].set_title(_list_simulation_names[j])
            axs2[i][j].set_xlabel('X1')
            axs2[i][j].set_ylabel('X2')
        axs2[i][0].set_ylabel(f"{math.ceil((i+1)/_n_fractions*100)}% of samples selected")    
        
    fig2.subplots_adjust(top=0.85)        
    fig2.savefig(f'{_path}/{_file_name}.png')    



def f_create_accuracy_chart(_df, _path, _col_x, _col_y, _hue='Simulation Type'):

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
        'Equal_Spread': '#db5f57', 
        'Dense_Areas_First': '#d3db57', 
        'Centroids_First': '#57db5f', 
        'Outliers_First': '#57d3db', 
        'BatchBALD': '#5f57db'
                    # '#9abbff', 
                    # '#c9d7f0'        
    }








    sns.set(rc={'figure.figsize':(15.7,8.27)})    
    sns.set_style('white')
    _chart = sns.lineplot(data=_temp_df_chart, 
                x=_col_x, 
                y=_col_y, 
                hue=_hue,
                palette=palette_hue_colors
                )

    figure = _chart.get_figure()
    figure.savefig(_path)








def f_model_accuracy(_args):

    _df, _model, _ordered_samples_id, _qtd_samples_to_train, _GPU_flag, _df_validation = _args
    
    _ordered_samples_id_temp = _ordered_samples_id[0:_qtd_samples_to_train+1]
    # print("LEN == ", len(_ordered_samples_id_temp))
    
    if _GPU_flag is True:
        _temp_X_columns = [x for x, mask in zip(_df.columns.values, _df.columns.str.startswith("X")) if mask]
        X_train = _df[_df['sample_id'].isin(_ordered_samples_id_temp)].loc[:,_temp_X_columns].astype('float32')
        y_train = _df[_df['sample_id'].isin(_ordered_samples_id_temp)].loc[:,'labels'].astype('float32')       
        X_test = _df.loc[:,_temp_X_columns].astype('float32')
        y_test = _df.loc[:,'labels'].astype('float32')
        X_validation = _df_validation.loc[:,_temp_X_columns].astype('float32')
        y_validation = _df_validation.loc[:,'labels'].astype('float32')             

    else:       
        # print("TPU")                                                          
        _temp_X_columns = list(_df.loc[:,_df.columns.str.startswith("X")].columns)                                                              
        X_train = _df[_df['sample_id'].isin(_ordered_samples_id_temp)].loc[:,_temp_X_columns]
        y_train = _df[_df['sample_id'].isin(_ordered_samples_id_temp)].loc[:,'labels']                    
        X_test = _df.loc[:,_temp_X_columns]
        y_test = _df.loc[:,'labels']
        X_validation = _df_validation.loc[:,_temp_X_columns]
        y_validation = _df_validation.loc[:,'labels']       
        # print("X_train .shape = ", X_train.shape)
        # print("X_test .shape = ", X_test.shape)


    try:                    
        _model.fit(X_train, y_train)                                    
        _score = _model.score(X_test, y_test)
        _score_validation = _model.score(X_validation, y_validation)
        # print("worked for..", _qtd_samples_to_train)
        # print("_score = ", _score)
        # print("\n\n")
        return _score, _score_validation
    
    except:                                         
        _score = 0
        _score_validation = 0
        print("entered i expection...", i)
        print("\n\n")
        return _score, _score_validation
