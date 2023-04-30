import time 
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.animation as animation





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
        


def generate_gif_chart_scatterplots(df, selected_samples, n_charts, chart_title, _path=None):
    # define the figure size and layout

    nrows = int(n_charts ** 0.5)
    ncols = int(n_charts / nrows)
    if n_charts > nrows * ncols:
        ncols += 1

    fig_size = (ncols * 6, nrows * 5)        
        
          
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size)

    _color_start = 'darkblue'
    _color_end = 'lightgray'

    # create a scatter plot for each chart
    for i, ax in enumerate(axes.flat):
        # calculate the number of selected samples for the current chart
        n_samples = int(len(selected_samples) * ((i + 1) / n_charts))
        samples = selected_samples[:n_samples]

        # create the scatter plot
        colors = [_color_start if s in samples else _color_end for s in df['sample_id']]
        ax.scatter(df['X1'], df['X2'], c=colors)
        ax.set_title(f'{int(((i + 1) / n_charts) * 100)}% of selected samples')

    # add the chart title to the top center of the figure
    fig.suptitle(chart_title, fontsize=16, y=1.05, x=0.5)

    # adjust the spacing between subplots
    fig.tight_layout()


    # save the figure as a .png file
    if _path != None:
        fig.savefig(f'{_path}/{chart_title}.png', dpi=300)        
    else:
        fig.savefig(f'{chart_title}.png', dpi=300)
        

    # create a scatter plot with all data points
    fig, ax = plt.subplots()
    sc = ax.scatter(df['X1'], df['X2'], c='gray')

    # define the update function for the animation
    def update(frame):
        colors = [_color_start if s in selected_samples[:frame+1] else _color_end for s in df['sample_id']]
        sc.set_color(colors)

        # add the chart title to the top center of the figure
        ax.set_title(chart_title, fontsize=16, y=1.05, x=0.5)

        return sc,

    # create the animation object
    ani = animation.FuncAnimation(fig, update, frames=len(selected_samples), interval=1000, blit=True)

    # save the animation as a .gif file
    if _path != None:        
        ani.save(f'{_path}/{chart_title}.gif', writer='imagemagick', fps=0.5)    
    else:
        ani.save(f'{chart_title}.gif', writer='imagemagick', fps=0.5)    

    # show the figure
    plt.show()


def f_create_chart(_df, _path, _col_x ='# Samples Evaluated/Interaction Number', _col_y='Accuracy', _hue='Simulation Type'):

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

