import numpy
import itertools
import pandas as pd
import random
import warnings


'''
    This file contains class object for RL fitting
'''

#####################################################RL FUNCTIONS###########################################################
class RL_mod(object):
    def __init__(self, animal_id, choices, outcomes):
        self.id = animal_id
        self.choices = numpy.array(choices).astype('int')
        self.outcomes = numpy.array(outcomes).astype('int')
    
    def make_selection(self, option_values, beta=3):
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
    
    def update_value_fq(self, qs, a, o, a_1, a_2, k_1, k_2):
        ''' Updates action value (q) for both possible choices according to a Q-Learning model with differential forgetting
            based on a given action and outcome in trial T. 
        :param qs: Array. Values for both choices at trial T-1. Left side value: index 0, Right side: 1. 
        :param a: integer. Side chosen on trial T. Left: 0, Right: 1. 
        :param o: integer. Outcome of trial T. Reward: 1. Nothing: 0. 
        :param a_1: float. alpha1. Forgetting rate for chosen action. High=recency bias. 
        :param a_2: float. alpha2. Forgetting rate for non-selected action. 
        :param k_1: float. kappa1. Strength of reward on action value. 
        :param k_2: float. kappa2. Strength of omission on action value. 
        :return new_qs: Array. Updated action values. 
        '''
        new_qs = qs.astype('float')
        for i in [0, 1]:
            qi = new_qs[i]
            if a==i:
                if o==1:
                    new_qs[i] = ((1-a_1) * qi) + (a_1*k_1)
                else:
                    new_qs[i] = (1-a_1)*qi - (a_1*k_2)
            else:
                new_qs[i] = (1-a_2)*qi
        return new_qs  
    
    def calc_Qs(self, params, return_qs = False):
        a1, a2, k1, k2, b = params
        values = numpy.array([0, 0])
        PP = []
        value_history = numpy.zeros([self.choices.size, 2])
        for t, c, o in zip(range(self.choices.size), self.choices, self.outcomes):
            mod_choice, p = self.make_selection(values, b)
            PP.append(p[c])

            values = self.update_value_fq(values, c, o, a1, a2, k1, k2)
            value_history[t] = values
        if return_qs:
            return value_history, PP
        else:
            return -1*numpy.log(PP).sum()


def update_value(v_last, r_last, alpha=0.1):
    warnings.warn("DeprecationWarning: Use class RL_Mod for fitting. Will raise AttributeError in v0.1a5.")
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
    warnings.warn("DeprecationWarning: Use class RL_Mod for fitting. Will raise AttributeError in v0.1a5.")
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
    warnings.warn("DeprecationWarning: Use class RL_Mod for fitting. Will raise AttributeError in v0.1a5.")
    '''
        Performs a grid search using update_value and make_selection. 
        :input alpha_range: The range of alpha values to test (used in update_value). 
        :input beta_range: The range of beta values to test (used in make_selection).
        :input choice_history: A binary list in which 0 corresponds to a left-sided choice and 1 to a right-sided choice.
                               # Will produce error if dtype!=int.
        :input outcome_history: A binary list in which 0 corresponds to a reward omission and 1 to a reward delivery.
        :input inital_values: Intial values to use for each iteration of the grid search. 
        :return search_grid: A DataFrame with index: alpha_range and columns: beta_range.
                             Contains the logsum of choice probabilities generated by each combination of 
                             alpha and beta. Values will be negative, with lower (i.e. larger magnitude) ones indicating
                             a worse fit. Perfect fit = 0. 
    '''
    search_grid = pd.DataFrame(index=alpha_range, columns = beta_range)
    for a, b in itertools.product(alpha_range, beta_range):
        values = initial_values.copy()
        PP = []
        for t, (c, o) in enumerate(zip(choice_history, outcome_history)):
            mod_choice, p = make_selection(values, beta=b) #Choose
            PP.append(p[c]) # Record P(mod_choice). This is why dtype of choice history MUST be integer.
            values[c] = update_value(values[c], o, alpha=a) # Update value of chosen otpion based on outcome
        
        # Calculate likelihood of all choices given the outcomes giving the current alpha,beta.
        search_grid.loc[a, b] = numpy.log(PP).sum()
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
