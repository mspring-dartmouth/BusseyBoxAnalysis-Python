import pandas as pd
import numpy as np
import itertools
import re


'''
    Functions that are used by other modules in this package. 
'''
#####################################################UTILITY FUNCTIONS###########################################################
def moving_average(a, n=5, pad_leading_zeros=True) :
    '''Calculates a moving average on single dimensional input using order "n".
       :param a: single dimensional input on which to calculate average.
       :param n: desired # of points to use in calculation of moving average. Default = 5.
       :param pad_leading_zeros: determines whether zeros will be added to front of output signal so that len(ret) == len(a); else len(ret) == len(a) - n
       :return ret: The moving average.
    '''
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    if pad_leading_zeros:
        # Pads  with leading zeros to avoid changing array size
        ret = np.insert(ret, 0, np.zeros(n-1))
    else:
        pass
    return ret

def enumerate2(iterable, start=0, step=1):
    '''Identical to built-in enumerate, but enables variation in start and step.
       :param iterable: an iterable which will provide the primary outputs in the loop
       :param start: value at which to begin enumeration
       :param step: amount by which secondary value will increase with each iteration.
       :return None: This is a wrapper function. 
    '''
    for x in iterable:
        yield (start, x)
        start += step

def enumerated_product(*args):
    ''' Similar to enumerate, but performs the iteration across combinations of multiple iterables. 
        Each iteration yields a tuple containing two tuples. The first contains the indices of the values in the second, 
        while the second contains the actual iteration across the iterables provided to args '''
    yield from zip(itertools.product(*(range(len(x)) for x in args)), itertools.product(*args))


'''
    Functions used for reading in raw Bussey-Box generated data files. 
'''
#####################################################FILE I/O FUNCTIONS###########################################################
def read_raw_file(input_file):
    '''
        DOCSTRING goes here!
    '''
    first_col = pd.read_csv(input_file, usecols=['Name'], nrows=50, skip_blank_lines=False)
    data_start_idx = first_col[first_col.Name=='Evnt_Time'].index[0]   
    
    session_info = pd.read_csv(input_file, nrows=data_start_idx-1, index_col=0, on_bad_lines='skip') # This will only cause problems if Animal ID, Date, or Environment are bad.  
    mouse_id = session_info.loc['Animal ID', 'Value']
    date = re.search('\d{1,2}/\d{1,2}/\d{4}', session_info.loc['Date/Time', 'Value']).group(0)
    box = session_info.loc['Environment'].values

    session_data = pd.Series(index = ['id', 'date', 'box'],
                             data = [mouse_id, date, box])
    raw_data_df = pd.read_csv(input_file, skiprows=data_start_idx+1)
    return raw_data_df, session_data

def batch_read_files(list_of_paths, use_date_as_key = True, mixup_fixer = None):
    '''
        DOCSTRING goes here!
    '''
    try:
        box_dict, id_switcher = mixup_fixer
        swap_ids = True
    except TypeError:
        swap_ids = False

    df_dict = {}
    for file in list_of_paths:
        data, sess_info = read_raw_file(file)
        mouse_id, date, box = sess_info[['id', 'date', 'box']]
        
        if swap_ids and box != box_dict[mouse_id]:
            print(f'{mouse_id}--->{id_switcher[mouse_id]} for {date}')
            mouse_id = id_switcher[mouse_id]

        if mouse_id not in df_dict.keys():
            df_dict[mouse_id] = {} if use_date_as_key else data
               
        if use_date_as_key:
            df_dict[mouse_id][date] = data

    return df_dict

#####################################################BASE DATA EXTRACTION FUNCTIONS###########################################################
def display_all_variables(raw_input_dataframe):
    '''
        DOCSTRING!!
    '''
    print(set(raw_input_dataframe.Item_Name))
    return list(set(raw_input_dataframe.Item_Name))

def get_final_values(raw_input_dataframe, metrics = ['_Trial_Counter']):
    '''
        What does this do?
        :param raw_input_dataframe: asdfasd
        :param metrics: asas
        :return out_series: asasdfas
    '''
    out_series = pd.Series(index=metrics)
    for metric in metrics:
        metric_idx = raw_input_dataframe[raw_input_dataframe.Item_Name==metric].index
        measure = raw_input_dataframe.loc[metric_idx[-1], 'Arg1_Value']
        out_series.loc[metric] = measure
    return out_series

def concatenate_values_over_days(animal_dictionary, dates, outer_metrics = ['_Trial_Counter']):
    '''
        What does this do?
        :param animal_dictionary: asdfasd
        :param dates: asas
        :param outer_metrics: asdfasl;dgkj
        :return cross_day_output: asasdfas
    '''
    multi_index = pd.MultiIndex.from_product([range(len(dates)), outer_metrics])
    cross_day_output = pd.Series(index = multi_index)
    for day, date in enumerate(dates):
        single_day_performance =  get_final_values(animal_dictionary[date], metrics=outer_metrics)
        cross_day_output.loc[(day, outer_metrics)] = single_day_performance.values
    return cross_day_output

def extract_timestamps(raw_input_dataframe, target, wide=True):
    '''
       DOCSTRING
    '''

    # Target may be either a single value or a list
    if isinstance(target, str):
        target = [target] # Turn it into a list for iteration

    max_len = raw_input_dataframe.loc[raw_input_dataframe.Item_Name.isin(target)].shape[0]
    ts_df = pd.DataFrame(index=range(max_len), columns=target)

    for t in target:
        evnt_ts = raw_input_dataframe.loc[raw_input_dataframe.Item_Name==t, 'Evnt_Time'].values
        ts_df.loc[range(evnt_ts.size), t] = evnt_ts

    ts_df.dropna(how='all', axis=0, inplace=True)

    if wide:
        return ts_df
    else:
        long_df = pd.DataFrame(index = range(ts_df.size), columns = ['Time', 'Label'])
        idx_start = 0
        for col in ts_df.columns:
            real_numbers = ts_df.loc[:, col].dropna(how='any', axis=0)
            idx_end = (idx_start + real_numbers.size)-1
            long_df.loc[idx_start:idx_end, 'Time'] = real_numbers.values
            long_df.loc[idx_start:idx_end, 'Label'] = col
            idx_start=idx_end+1

        long_df.dropna(how='any', axis=0, inplace=True)
        ordered_events = pd.Series(index=long_df.Time.astype('float'), data=long_df.Label.values)
        return ordered_events.sort_index()
