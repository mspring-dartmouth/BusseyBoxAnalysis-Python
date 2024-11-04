import numpy as np
import pandas as pd 

from busboxanalysis import get_final_values


beam_names = ['FIRBeam #1', 'RightFIRBeam #1']

def retrieve_cs_display_times(raw_input_dataframe):
    # Determine CS+ side:
    cs_plus = get_final_values(raw_input_dataframe, ['CS_Plus']).values[0]
    cs_minus = 3-cs_plus

    # Pull plus and minus on and off times
    cs_times = {'plus': {}, 'minus': {}}

    for cs_label, side in zip(['plus', 'minus'], [cs_plus, cs_minus]):
        for event, code in zip(['on', 'off'], ['Images', 'Background']):
            cs_times[cs_label][event] = raw_input_dataframe[(raw_input_dataframe.Alias_Name==code)&(raw_input_dataframe.Arg1_Value==side)].Evnt_Time.values

        cs_times[cs_label]['off'] = cs_times[cs_label]['off'][1:] # The first background image displayed is at session initiation, not cs offset. 

    return cs_times, {'plus': cs_plus, 'minus': cs_minus}

def summarize_trial_behavior(trial_slice, slice_borders, target_item_name = 'Tray #1'):
    entries = trial_slice[trial_slice.Evnt_ID==38].loc[:, 'Evnt_Time'].values
    exits = trial_slice[trial_slice.Evnt_ID==39].loc[:, 'Evnt_Time'].values
    
    on, off = slice_borders

    # If there are no events, skip to the end.
    if exits.size + entries.size == 0:
        pass

    # If either event is empty, filling in the boundary timestamp as done below will not work.
    elif min([exits.size, entries.size]) == 0:
        if exits.size == 0:
            exits = np.array([off]) # One unended entry gets capped with the end of the CS.
        else:
            entries = np.array([on]) # One headless exit is considered starting at CS on. 
    else:
        if exits[0] < entries[0]: # First exit is before first entry
            entries = np.insert(entries, 0, on)
        if entries[-1] > exits[-1]: # Last entry is after last exit
            exits = np.append(exits, off)

        if entries.size != exits.size:
            start_stop_id_codes = pd.Series(index = np.concatenate([entries, exits]),
                                            data = np.concatenate([np.zeros(entries.size), np.ones(exits.size)]))
            start_stop_id_codes.sort_index(inplace=True)
            dupes = []
            for i, j in zip(start_stop_id_codes.index[:-1], start_stop_id_codes.index[1:]):
                if start_stop_id_codes.loc[i] == start_stop_id_codes.loc[j]:
                    code = start_stop_id_codes.loc[i]
                    dupes.append(i if code==0 else j) # If initiations are duplicated, drop the first one. If it's a cessation, drop the second
            start_stop_id_codes.drop(dupes, inplace=True)
            entries = start_stop_id_codes[start_stop_id_codes==0].index.values
            exits = start_stop_id_codes[start_stop_id_codes==1].index.values
    return(entries, exits)



# Deprecated counts based on beam crossing for Sign Tracking
# def summarize_cs_responding(raw_input_dataframe, cs_times, cs_assignments):
#     responding_summary = {'plus': {}, 'minus': {}}
#     for cs in ['plus', 'minus']:
#         responding_summary[cs]['goaltrack'] = pd.DataFrame(index = range(cs_times[cs]['on'].size), columns=['Count', 'Mean_Dur'])
#         responding_summary[cs]['signtrack'] = pd.DataFrame(index = range(cs_times[cs]['on'].size), columns=['Count', 'Mean_Dur'])
#         for on, off, cs_num in zip(cs_times[cs]['on'], cs_times[cs]['off'], responding_summary[cs]['goaltrack'].index):
#             cs_presentation_slice = raw_input_dataframe[(raw_input_dataframe.Evnt_Time>=on)&(raw_input_dataframe.Evnt_Time<=off)]
#             beam_name = beam_names[int(cs_assignments[cs])-1] # Pull cs side code from cs_assignments dictionary and convert to index
#             for behavior, input_name in zip(['goaltrack', 'signtrack'], ['Tray #1', beam_name]):
#                 behavior_slice = cs_presentation_slice[cs_presentation_slice.Item_Name==input_name]
#                 entries, exits = summarize_trial_behavior(behavior_slice, (on, off), input_name)
#                 responding_summary[cs][behavior].loc[cs_num, 'Count'] = entries.size
#                 responding_summary[cs][behavior].loc[cs_num, 'Mean_Dur'] = np.sum(exits-entries)
#     return responding_summary



# Count Sign Tracking based on touch counts
def summarize_cs_responding(raw_input_dataframe, cs_times, cs_assignments):
    responding_summary = {'plus': {}, 'minus': {}}
    
    simple_track = create_touch_tracking(raw_input_dataframe)
    for cs in ['plus', 'minus']:
        responding_summary[cs]['goaltrack'] = pd.DataFrame(index = range(cs_times[cs]['on'].size), columns=['Count', 'Mean_Dur'])
        responding_summary[cs]['signtrack'] = pd.DataFrame(index = range(cs_times[cs]['on'].size), columns=['Count', 'Mean_Dur'])
        for on, off, cs_num in zip(cs_times[cs]['on'], cs_times[cs]['off'], responding_summary[cs]['goaltrack'].index):
            cs_presentation_slice = raw_input_dataframe[(raw_input_dataframe.Evnt_Time>=on)&(raw_input_dataframe.Evnt_Time<=off)]
            beam_name = beam_names[int(cs_assignments[cs])-1] # Pull cs side code from cs_assignments dictionary and convert to index
            for behavior, input_name in zip(['goaltrack', 'signtrack'], ['Tray #1', beam_name]):
                if behavior=='goaltrack':
                    behavior_slice = cs_presentation_slice[cs_presentation_slice.Item_Name==input_name]
                    entries, exits = summarize_trial_behavior(behavior_slice, (on, off), input_name)
                    responding_summary[cs][behavior].loc[cs_num, 'Count'] = entries.size
                    responding_summary[cs][behavior].loc[cs_num, 'Mean_Dur'] = np.sum(exits-entries)
                else:
                    touch_count = simple_track[(simple_track.Evnt_ID==31)&(simple_track.LastCrossed=='Left')&(simple_track.Evnt_Time>=on)&(simple_track.Evnt_Time<=off)].shape[0]
                    responding_summary[cs][behavior].loc[cs_num, 'Count'] = touch_count


    return responding_summary


# Auxilary function for determining which side a touch occurred on. 
def create_touch_tracking(raw_input_dataframe):
    # Create dataframe that contains touches and a LastCrossed
    simple_track = raw_input_dataframe[(raw_input_dataframe.Evnt_ID==31)|((raw_input_dataframe.Evnt_ID==38)&(raw_input_dataframe.Item_Name.isin(['FIRBeam #1', 'RightFIRBeam #1'])))]
    simple_track = simple_track.loc[:, ['Evnt_Time', 'Evnt_ID', 'Item_Name']]
    simple_track['LastCrossed'] = np.empty(simple_track.shape[0])
    simple_track.reset_index(inplace=True)
    simple_track.drop('index', axis=1, inplace=True)

    # Identify the last beam crossed for every tracked moment
    crossings = simple_track[simple_track.Evnt_ID==38].index
    for c_last, c_next in zip(crossings[:-1], crossings[1:]):
        simple_track.loc[c_last:c_next, 'LastCrossed'] = 'Left' if simple_track.loc[c_last, 'Item_Name'] =='FIRBeam #1' else 'Right'


    return simple_track
