import time 
import datetime
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from sklearn.datasets import make_blobs






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
#         df_subset['color'] = 'gray'
#         df_subset['fraction'] = ''
#         scatter_plot = axs[i].scatter(df_subset['X1'], df_subset['X2'], c=df_subset['color'])
#         scatter_plots.append(scatter_plot)
#         axs[i].set_title(_list_chart_titles[i])


#     def animate(i):
#         for j in range(num_simulations):
#             sample_ids = _list_selected_samples[j]
#             df_subset = _df.copy()
#             df_subset['color'] = 'gray'
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
#     _df['color'] = 'gray'
    
#     scatter_plot = axs[0].scatter(df_subset['X1'], df_subset['X2'], c='gray')
#     fig2, axs2 = plt.subplots(nrows=fractions, ncols=num_simulations, figsize=(4*num_simulations, 4*fractions), tight_layout=True)
#     i = 0
#     for i in range(fractions):
#         for j in range(num_simulations):
#             sample_ids = _list_selected_samples[j]
#             df_subset = _df.copy()
#             df_subset['color'] = 'gray'
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
#     _color_end = 'lightgray'

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
#     sc = ax.scatter(_df['X1'], _df['X2'], c='gray')

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
        df_subset['color'] = 'gray'
        df_subset['fraction'] = ''
        scatter_plot = axs[i].scatter(df_subset['X1'], df_subset['X2'], c=df_subset['color'])
        scatter_plots.append(scatter_plot)
        axs[i].set_title(_list_simulation_names[i])


    def animate(i):
        for j in range(num_simulations):
            sample_ids = _list_selected_samples[j]
            df_subset = _df_2D.copy()
            df_subset['color'] = 'gray'
            index = i % len(sample_ids)
            df_subset.loc[df_subset['sample_id'].isin(sample_ids[:index+1]), 'color'] = 'blue'
            df_subset.loc[df_subset.index <= math.ceil((index+1)/len(sample_ids)*_n_fractions)*len(df_subset)/_n_fractions,'fraction'] = f"{math.ceil((index+1)/len(sample_ids)*100)}%"
            scatter_plots[j].set_color(df_subset['color'])
            axs[j].set_xlabel('fraction')

        # axs[0].table(cellText=[df_subset['fraction'].unique()], loc='bottom', cellLoc='center')
        # axs[0].axis('off')

    ani = animation.FuncAnimation(fig, animate, frames=len(_list_selected_samples[0]), interval=100, repeat=True)
    ani.save(f'{_path}/{_file_name}.gif', writer='imagemagick', fps=_fps)    
    
    _df_2D['color'] = 'gray'
    
    scatter_plot = axs[0].scatter(df_subset['X1'], df_subset['X2'], c='gray')
    fig2, axs2 = plt.subplots(nrows=_n_fractions, ncols=num_simulations, figsize=(4*num_simulations, 4*_n_fractions), tight_layout=True)
    i = 0
    for i in range(_n_fractions):
        for j in range(num_simulations):
            sample_ids = _list_selected_samples[j]
            df_subset = _df_2D.copy()
            df_subset['color'] = 'gray'
            df_subset['fraction'] = ''
            index = math.ceil((i+1)/_n_fractions*len(sample_ids))
            df_subset.loc[df_subset['sample_id'].isin(sample_ids[:index]), 'color'] = 'blue'
            axs2[i][j].scatter(df_subset['X1'], df_subset['X2'], c=df_subset['color'])
            axs2[i][j].set_title(_list_simulation_names[j])
            axs2[i][j].set_xlabel('X1')
            axs2[i][j].set_ylabel('X2')
        axs2[i][0].set_ylabel(f"{math.ceil((i+1)/_n_fractions*100)}% of samples selected")    
        
    fig.subplots_adjust(top=0.85)    
    plt.savefig(f'{_path}/{_file_name}.png')    



def f_create_accuracy_chart(_df, _path, _col_x ='# Samples Evaluated/Interaction Number', _col_y='Accuracy', _hue='Simulation Type'):

    _list_cols = [_col_x, _col_y, _hue]
    
    _temp_df_chart = _df[_list_cols]    
    _temp_df_chart = _temp_df_chart.reset_index(drop=True)                    

    
    #[TO-DO] change this to be dynamic... 
    # palette = sns.color_palette("mako", len(_list_models))                        
    palette = ['#F22B00', '#40498e', '#357ba3', '#38aaac', '#79d6ae']    

    sns.set(rc={'figure.figsize':(15.7,8.27)})    
    _chart = sns.lineplot(data=_temp_df_chart, 
                x=_col_x, 
                y=_col_y, 
                hue=_hue,
                palette=palette
                )

    figure = _chart.get_figure()
    figure.savefig(_path)

