import numpy
import itertools
import pandas as pd
import random


'''
    This file contains function for RL fitting
'''

# TODO:
    # Change value and choice function names to be specific to what the current ones do. 
    # Change grid search to accept value and choice functions for use in search as arguments. 

#####################################################RL FUNCTIONS###########################################################
def update_value(v_last, r_last, alpha=0.1):
    '''
        The R-W value function. 
        :param v_last: the expected value of an option prior to the last time it was selected.
        :param r_last: the reward outcome of the same option the last time it was selected.
        :param alpha: the learning rate. That is, how much of the prediction error should be applied to updating an option's value
        :return v: the expected value of an option in the future. 
    '''
    pe = alpha*(r_last-v_last)
    v = v_last + pe
    return v

def make_selection(option_values, beta=3):
    '''
        A choice function for 2 options based on a soft-max operation.
        :param option_values: an array containing the expected values of the two options, [0] and [1].
        :param beta: temperature of the soft-max operation (float). 
                     High values produce greater exploitation of preferred options, while lower values produce exploration.
        :return x, p: return a choice 0 or 1, and the probability associated with choosing either option (ordered 0, 1).
    '''
    ev = numpy.exp(beta*option_values)
    sev = sum(numpy.exp(beta*option_values))
    p = ev/sev
    
    if random.random()<p[0]:
        return 0, p
    else:
        return 1, p

def grid_search(alpha_range, beta_range, choice_history, outcome_history, initial_values = numpy.array([0.5, 0.5])):
    '''
        DOCSTRING goes here!
    '''
    search_grid = pd.DataFrame(index=alpha_range, columns = beta_range)
    for a, b in itertools.product(alpha_range, beta_range):
        values = initial_values.copy()
        PP = []
        for t, (c, o) in enumerate(zip(choice_history, outcome_history)):
            mod_choice, p = make_selection(values, beta=b) #Choose
            PP.append(p[c]) # Record P(mod_choice)
            values[c] = update_value(values[c], o, alpha=a) # Update value of chosen otpion based on outcome
        
        # Calculate likelihood of all choices given the outcomes giving the current alpha,beta.
        search_grid.loc[a, b] = numpy.prod(PP)
    return search_grid


# Currently unused, given that gamma has been removed from make_selection().
# def grid_search_3param(alpha_range, beta_range, gamma_range, choice_history, outcome_history, initial_values = numpy.array([0.5, 0.5])):
#     search_grid = numpy.empty([gamma_range.size, alpha_range.size, beta_range.size])
    
#     pd.DataFrame(index=alpha_range, columns = beta_range)
    
#     for idx, params in enumerated_product(gamma_range, alpha_range, beta_range):
#         g, a, b = params
#         values = initial_values.copy()
#         PP = []
#         for c, o in zip(choice_history, outcome_history):
#             mod_choice, p = make_selection(values, beta=b, gamma=g)
            
#             PP.append(p[c])

#             # Update values based on outcome
#             values[c] = update_value(values[c], o, alpha=a)
#         # Once you have all your choice probabilities, calculate likelihood

#         search_grid[idx] = numpy.prod(PP)
    
#     return search_grid
