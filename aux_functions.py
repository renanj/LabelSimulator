import time 
import datetime



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


def f_print(_string, _level=0, _write_option=False):


    _levels_allowed = [0,1,2,3,4,5,6,7,8,9,10,11,12]

    _string = f_saved_strings(_string)

    if _level not in _levels_allowed:
        return None
    else:
        if _write_option == False:
            return print(('  ' * _level) + _string)

        else: 
            new_string = ('\t' * _level) + _string + '\n'
            return new_string



def f_write(_string, _level=0):

    _levels_allowed = [0,1,2,3,4,5,6,7,8,9,10,11,12]

    _string = f_saved_strings(_string)

    if _string is None:
      _string = ''

    if _level not in _levels_allowed:                
        f.write('')
        f.flush                
    else:
        f.write(_string)
        f.flush()

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



def f_model_accuracy(_args):

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    _df, _model, _ordered_samples_id, _qtd_samples_to_train, _GPU_flag = _args
    
    _ordered_samples_id_temp = _ordered_samples_id[0:_qtd_samples_to_train+1]
    # print("LEN == ", len(_ordered_samples_id_temp))
    
    if _GPU_flag is True:
        _temp_X_columns = [x for x, mask in zip(_df.columns.values, _df.columns.str.startswith("X")) if mask]
        X_train = _df[_df['sample_id'].isin(_ordered_samples_id_temp)].loc[:,_temp_X_columns].astype('float32')
        y_train = _df[_df['sample_id'].isin(_ordered_samples_id_temp)].loc[:,'labels'].astype('float32')       
        X_test = _df.loc[:,_temp_X_columns].astype('float32')
        y_test = _df.loc[:,'labels'].astype('float32')

    else:                                                                    
        _temp_X_columns = list(_df.loc[:,_df.columns.str.startswith("X")].columns)                                                                
        X_train = _df[_df['sample_id'].isin(_ordered_samples_id_temp)].loc[:,_temp_X_columns]
        y_train = _df[_df['sample_id'].isin(_ordered_samples_id_temp)].loc[:,'labels']                      
        X_test = df.loc[:,_temp_X_columns]
        y_test = df.loc[:,'labels']


    try:      
        with parallel_backend('multiprocessing'):                              
            _model.fit(X_train, y_train)                                    
            _score = _model.score(X_test, y_test)
            #print("worked for..", _qtd_samples_to_train)
            return _score
        
    except:                                            
        _score = 0
        #print("entered i expection...")
        return _score
