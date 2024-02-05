import pandas as pd
import numpy as np
import natsort
from busboxanalysis import get_final_values, extract_timestamps


# Turn this into a function to  

def summarize_responses(raw_input_dataframe):
    '''
        Summarizes responses into each window for a given animal on a given day.
        :input raw_input_dataframe: A dataframe output by busboxanalysis.read_raw_file/busboxanalysis.batch_read_files
        :return tuple: This function outputs a tuple of the following, in order:
        :return response_distribution: a pandas Dataframe with a multi-index: response category X response window 
                                       and values: no. of responses
        :return prem_response_count: Integer count of premature responses. 
        :return total_trials: Integer count of trials completed. 
    '''
    m_index = pd.MultiIndex.from_product([['Correct', 'Incorrect', 'Omission'], range(1, 6)])
    m_index.set_names(['Response', 'Window'], inplace=True)
    response_distribution = pd.DataFrame(index=m_index, columns = ['N_Responses'])

    for r_type in ['Correct', 'Incorrect', 'Omission']:
        for window in range(1, 6):
            response_distribution.loc[(r_type, window), 'N_Responses'] = raw_input_dataframe[raw_input_dataframe.Item_Name==f'{r_type} To {window}'].shape[0]
    summary_vals = get_final_values(raw_input_dataframe, metrics=['Premature_Response_Counter', '_Trial_Counter'])
    prem_response_count, total_trials = summary_vals.loc[['Premature_Response_Counter', '_Trial_Counter']]

    return response_distribution, prem_response_count, total_trials






def retrieve_touch_times(raw_input_dataframe, return_trial_dict = True, return_latencies = True):
    '''
        Retrieves and categorizes touch times for a single dataframe.
        :input raw_input_dataframe: A dataframe output by busboxanalysis.read_raw_file/busboxanalysis.batch_read_files
        :input return_trial_dict: Toggles whether or not to return trial_touch_dict.
        :input return_latencies: Toggles whether to return latency_df
        :return touches: A pandas DataFrame with index: indices of touches, columns: Time of touch and Type of Touch.
        :return trial_touch_dict: A dictionary that contains all touch types and times for each trial:
                                  Key: trial number, Value: dict{Key: Touch category, Value: array of timestamps}
        :return latency_df: A DataFrame with index: trial number, 
                            and columns [Start Time, Response Time, Response Latency, and Response Type].
                            IGNORES PREMATURE RESPONSES
    '''

    # Pull indices of touch events. 
    touch_idxs = raw_input_dataframe[(raw_input_dataframe.Evnt_Name == 'Touch Down Event')].index

    # Create a dataframe to store times and touch categories. 
    touches = pd.DataFrame(index=range(len(touch_idxs)), columns = ['Time', 'TouchType'])
    touches.loc[:, 'Time'] = raw_input_dataframe.loc[touch_idxs, ['Evnt_Time']].values


    # Iterate over all touch events and categorize them.
    for df_idx, touch in enumerate(touch_idxs):
        touch_code = raw_input_dataframe.loc[touch, 'Group_ID']

        # These first three can be categorized based on the Group_ID alone. 

        # I'm not calling "Timeout" touches premature. Only Premature Touches
        if touch_code == 4:
            t_type = 'Premature'
        elif touch_code == 5:
            t_type = 'TO_Touch'
        # 13 are touches that occur during a squishy in-between window
        elif touch_code == 13:
            t_type = 'Unclassified'

        # 7 or 11 are touches during properly completed trials. 
        # These are the only reason to have a loop do this. 
        # Group_ID==7 can be either correct or incorrect, and 11 is a perseverative flavor of the same. 
        # The next line in the DF is necessary to decode. 
        elif touch_code in [7, 11]:    
            if touch_code == 7:
                if 'Incorrect' in raw_input_dataframe.loc[touch+1, 'Item_Name']:
                    t_type = 'IncorrectResponse'
                elif 'Correct' in raw_input_dataframe.loc[touch+1, 'Item_Name']:
                    t_type = 'CorrectResponse'
            elif touch_code == 11:
                if raw_input_dataframe.loc[touch+1, 'Item_Name'] == 'Perseverative Incorrect Response ':
                    t_type = 'IncorrectResponsePerseveration'
                elif raw_input_dataframe.loc[touch+1, 'Item_Name'] == 'Perseverative Response to Correct ':
                    t_type = 'CorrectResponsePerseveration'

        # At the end of the nested for-loops, record the category. 
        touches.loc[df_idx, 'TouchType'] = t_type

    # If return_latencies==True, then the whole function needs to run.
    if return_latencies:
        return_trial_dict = True    
    # Exit if neither trial_dict nor latency_df were requested.
    if return_trial_dict == False:
        return touches    
    #######################################################################################################################################################
    # Separate events by which trial they occurred during. 

    # Set the start of a new trial as the beginning of the ITI. 
    iti_start_times = extract_timestamps(raw_input_dataframe, 'Start ITI', wide=False).index

    t_types = ['CorrectResponse', 'CorrectResponsePerseveration', 'IncorrectResponse', 'IncorrectResponsePerseveration', 'Premature', 'Unclassified', 'TO_Touch']
    trial_touch_dict = {} # Key: trial number, Value: dict{Key: Touch category, Value: array of timestamps}
    for t_num in range(len(iti_start_times)):
        trial_touch_dict[t_num] = {}
    
    for t_type in t_types:
        # Iterating over touch types, pull all timestamps for given category.
        t_times = touches.loc[touches.TouchType==t_type, 'Time']

        # Use end of trial as boundary for final trial. 
        for t_num, borders in enumerate(zip(iti_start_times, np.append(iti_start_times[1:], raw_input_dataframe.Evnt_Time.max()))):
            t_start, t_end = borders
            trial_touch_dict[t_num][t_type] = t_times[(t_times>=t_start)&(t_times<t_end)].values

    # Exit if latency_df wasn't requested.
    if return_latencies == False:
        return touches, trial_touch_dict
    #######################################################################################################################################################
    # Create a dataframe that has each row as a trial number, and includes columns [Start Time, Response Time, Response Latency, and Response Type.]
    # THIS IGNORES PREMATURE RESPONSES

    latency_df = pd.DataFrame(index=trial_touch_dict.keys(), columns=['T_Start', 'Touch_Time', 'Latency', 'Response_Type'])
    
    for idx in latency_df.index:
        latency_df.loc[idx, 'T_Start'] = iti_start_times[idx]
        if trial_touch_dict[idx]['CorrectResponse'].size + trial_touch_dict[idx]['IncorrectResponse'].size == 0:
            latency_df.loc[idx, 'Response_Type'] = 'Omission'
            continue
        try:
            response_time = trial_touch_dict[idx]['CorrectResponse'][0]
            response_type = 'Correct'
        except IndexError:
            response_time = trial_touch_dict[idx]['IncorrectResponse'][0]
            response_type = 'Incorrect'
            
        latency_df.loc[idx, 'Touch_Time'] = response_time
        latency_df.loc[idx, 'Latency'] = response_time - iti_start_times[idx]
        latency_df.loc[idx, 'Response_Type'] = response_type


    return touches, trial_touch_dict, latency_df