from busboxanalysis import moving_average 
import numpy as np

'''
    This file holds functions specifically for summarizing data from Bussey-Box raw files for the Free Foraging task. 
    (i.e. ProbabilisticForagingTask_[ChoiceTraining/IntermediateTraining/Main])
'''

def summarize_right_side_reward_probability(raw_input_dataframe, fudge = 0, absolute_value = True):
    '''
       Reads the reward probability associated with a right-sided choice over the course of a behavioral session in  session.
       :param raw_input_dataframe: a pandas dataframe created from a single Bussey-Output .csv containing raw session data.
       :param fudge: number of values to manually add to final block if needed.
       :param absolute_value: determine whether to return the absolute or relative reward values associated with a right-side choice.
       :return right_side_history: a list containing the reward probabilities associated with a right-sided choice for every trial in a session.
    '''
    
    high_prob = raw_input_dataframe[raw_input_dataframe.Item_Name=='HighProbability'].loc[:, 'Arg1_Value'].values[0] # High probability used throughout the session.
    raw_block_lens = raw_input_dataframe[raw_input_dataframe.Item_Name=='BlockLen'].loc[:, 'Arg1_Value'].values # Block lengths over the entire session
    n_trials_final_block = raw_input_dataframe[(raw_input_dataframe.Item_Name=='Block_TrialNum')].loc[:, 'Arg1_Value'].values[-1] # Number of trials completed in the final block
    n_trials_final_block += fudge # This is used only when this function is called by concatenate_behavioral_days() and something doesn't match up.
    low_probs = raw_input_dataframe[raw_input_dataframe.Item_Name=='LowPSideValue'].loc[:, 'Arg1_Value'].values # Low probabilities used within each block
    high_prob_sides = raw_input_dataframe[raw_input_dataframe.Item_Name=='LargeRewardSide'].loc[:, 'Arg1_Value'].values # Side associated with higher probability of reward delivery in each block
    # In earlier versions of the ABET task, the probability was initialized using a default value, yielding an extra value in "low_probs"
    # In later versions, the probability is drawn at the start of the trial. 
    if len(raw_block_lens) < len(low_probs):
        low_probs = low_probs[1:]
        high_prob_sides = high_prob_sides[1:]
    # Iterate over blocks and fill right_side_history with the probabilities associated with right side responding for the number of trials in the block.
    right_side_history = [] 
    for block_len, high_side, low_prob in zip(raw_block_lens[:-1], high_prob_sides[:-1], low_probs[:-1]):
        right_side_block_prob = high_prob if high_side==2 else low_prob
        
        # If using relative value, set modification to left-side probability from current block
        mod = 0 if absolute_value==True else high_prob if high_side==1 else low_prob

        probs = np.ones(int(block_len+1)) * right_side_block_prob
        right_side_history.extend(probs-mod)
    # Tack on values for trials in final block
    last_block_right_side_prob = high_prob if high_prob_sides[-1] ==2 else low_probs[-1]
    mod = 0 if absolute_value==True else high_prob if high_prob_sides[-1]==1 else low_probs[-1]
    right_side_history.extend(np.ones(int(n_trials_final_block))*(last_block_right_side_prob-mod))
    
    return right_side_history

def summarize_relative_right_side_value(raw_input_dataframe, fudge = 0):
    raise NameError('This function has been deprecated. Use summarize_right_side_reward_probability with absolute_value=False instead.')

def summarize_behavior(raw_input_dataframe, return_absolute_for_right = True):
    '''
        Summarizes response history, outcome history, and calls "summarize_right_side_reward_probability" 
        or "summarize_relative_right_side_value" for a single animal on a single day.
        :param raw_input_dataframe: a pandas dataframe created from a single Bussey-Output .csv containing raw session data.
        :param return_absolute_for_right: Toggles whether to call summarize_right_side_reward_probability with absolute_value=True or absolute_value=False
        :return response history: A binary list in which 0 corresponds to a left-sided choice and 1 to a right-sided choice.
        :return outcome history: A binary list in which 0 corresponds to a reward omission and 1 to a reward delivery.
        :return right_side_history: A list containing the requested reward probabilities associated with a right-sided choice
    '''
    
    # Retrive the indices at which counters four possible responseXoutcome combinations were updated.
    response_types = ['LeftResponse_Rewarded_Counter', 'LeftResponse_NoReward_Counter', 'RightResponse_Rewarded_Counter', 'RightResponse_NoReward_Counter']
    response_idxs = []
    for r_t in response_types:
        response_idxs.extend(raw_input_dataframe[raw_input_dataframe.Item_Name==r_t].index[1:])
    
    # Sort the response categorizations by index to get a time-ordered list of all trial results.
    composite_history = raw_input_dataframe.loc[sorted(response_idxs), 'Item_Name']
    response_history = [int('Right' in x) for x in composite_history] # Create list of responses where 1=Right side choice and 0=Left side choice
    outcome_history = [int('Rewarded' in x) for x in composite_history] # Create list of outcomes where 1=Reward received and 0=No reward received
    
    # Retrieve the probability of a reward available on the righthand side.
    right_side_history = summarize_right_side_reward_probability(raw_input_dataframe, absolute_value=return_absolute_for_right)
    # If the number of probabilities does not match the number of trials completed, pad the last block. This is a workaround rather than a true fix. 
    if len(right_side_history) != len(composite_history): 
        fudge_factor = len(composite_history) - len(right_side_history)
        right_side_history = summarize_right_side_reward_probability(raw_input_dataframe, fudge=fudge_factor, absolute_value=return_absolute_for_right)
    
    return response_history, outcome_history, right_side_history

def concatenate_behavioral_days(animal_dictionary, dates, return_absolute_for_right = True, smooth_order=15):
    '''
        Iterates over a sequence of days for a single animal, calling "summarize_behavior" on each day and smoothing the output
        for more interpretable graphing.
        :param animal_dictionary: A dictionary containing key, value pairs of dates and input_dataframes, respectively.
        :param dates: a list of dates, in the same format at those in animal_dictionaries, over which to iterate.
        :param return_absolute_for_right: Toggles whether to call summarize_right_side_reward_probability or summarize_relative_right_side_value
        :param smooth_order: Desired # of points to use in calculation of moving average.
        :return trials: List of trial numbers 1- n_trials.
        :return choice_history history: A binary list in which 0 corresponds to a left-sided choice and 1 to a right-sided choice.
        :return outcome history: A binary list in which 0 corresponds to a reward omission and 1 to a reward delivery.
        :return block_history: A list containing the requested reward probabilities associated with a right-sided choice
    '''    
    choice_history = []
    outcome_history = []
    block_history = []
    for date in dates:
        func_return = summarize_behavior(animal_dictionary[date], return_absolute_for_right=return_absolute_for_right)
        choice_history.extend(moving_average(func_return[0], n=smooth_order)[smooth_order-1:])
        outcome_history.extend(moving_average(func_return[1], n=smooth_order)[smooth_order-1:])
        block_history.extend(func_return[2][smooth_order-1:])
    trials = np.arange(1, len(choice_history)+1)
    return trials, choice_history, outcome_history, block_history






# #####################################################RL FUNCTIONS###########################################################
# def update_value(v_last, r_last, alpha=0.1):
#     '''
#         The R-W value function. 
#         :param v_last: the expected value of an option prior to the last time it was selected.
#         :param r_last: the reward outcome of the same option the last time it was selected.
#         :param alpha: the learning rate. That is, how much of the prediction error should be applied to updating an option's value
#         :return v: the expected value of an option in the future. 
#     '''
#     pe = alpha*(r_last-v_last)
#     v = v_last + pe
#     return v

# def make_selection(option_values, beta=3):
#     '''
#         A choice function for 2 options based on a soft-max operation.
#         :param option_values: an array containing the expected values of the two options, [0] and [1].
#         :param beta: temperature of the soft-max operation (float). 
#                      High values produce greater exploitation of preferred options, while lower values produce exploration.
#         :return x, p: return a choice 0 or 1, and the probability associated with choosing either option (ordered 0, 1).
#     '''
#     ev = np.exp(beta*option_values)
#     sev = sum(np.exp(beta*option_values))
#     p = ev/sev
    
#     if random.random()<p[0]:
#         return 0, p
#     else:
#         return 1, p

# def grid_search(alpha_range, beta_range, choice_history, outcome_history, initial_values = np.array([0.5, 0.5])):
#     '''
#         DOCSTRING goes here!
#     '''
#     search_grid = pd.DataFrame(index=alpha_range, columns = beta_range)
#     for a, b in itertools.product(alpha_range, beta_range):
#         values = initial_values.copy()
#         PP = []
#         for t, (c, o) in enumerate(zip(choice_history, outcome_history)):
#             mod_choice, p = make_selection(values, beta=b) #Choose
#             PP.append(p[c]) # Record P(mod_choice)
#             values[c] = update_value(values[c], o, alpha=a) # Update value of chosen otpion based on outcome
        
#         # Calculate likelihood of all choices given the outcomes giving the current alpha,beta.
#         search_grid.loc[a, b] = np.prod(PP)
#     return search_grid