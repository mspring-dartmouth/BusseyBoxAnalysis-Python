import numpy
import random


'''
    This file contains models for updating values, making selections, and generating choice probabilities
'''

#####################################################RL FUNCTIONS###########################################################

# Built-in Models

# Q-LEARNING MODELS
def update_value_prediction_error(qs, a, o, model_params):
    ''' Updates action value (q) for both last chosen value according to prediction error. 
    :param qs: Array. Values for both choices at trial T-1. Left side value: index 0, Right side: 1. 
    :param a: integer. Side chosen on trial T. Left: 0, Right: 1. 
    :param o: integer. Outcome of trial T. Reward: 1. Nothing: 0. 
    :param model_params: tuple containing the following values:
        alpha: float. alpha. Learning rate for chosen action. High=recency bias. 
    :return new_qs: Array. Updated action values. 
    '''
    alpha, = model_params
    pe = alpha * (o - qs[a])
    qs[a] = qs[a] + pe

    return qs, model_params


def update_value_dfq_static(qs, a, o, model_params):
    ''' Updates action value (q) for both possible choices according to a Q-Learning model with differential forgetting
        based on a given action and outcome in trial T. 
    :param qs: Array. Values for both choices at trial T-1. Left side value: index 0, Right side: 1. 
    :param a: integer. Side chosen on trial T. Left: 0, Right: 1. 
    :param o: integer. Outcome of trial T. Reward: 1. Nothing: 0. 
    :param model_params: tuple containing the following values:
        a_1: float. alpha1. Forgetting rate for chosen action. High=recency bias. 
        a_2: float. alpha2. Forgetting rate for non-selected action. 
        k_1: float. kappa1. Strength of reward on action value. 
        k_2: float. kappa2. Strength of omission on action value. 
    :return new_qs: Array. Updated action values. 
    '''
    a_1, a_2, k_1, k_2 = model_params
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
    return new_qs, model_params  

def update_value_fq_static(qs, a, o, model_params):
    ''' Updates action value (q) for both possible choices according to a Q-Learning model with CONSTANT forgetting
        based on a given action and outcome in trial T. This is equivalent to dfq_static with a_1=a_2. 
    :param qs: Array. Values for both choices at trial T-1. Left side value: index 0, Right side: 1. 
    :param a: integer. Side chosen on trial T. Left: 0, Right: 1. 
    :param o: integer. Outcome of trial T. Reward: 1. Nothing: 0. 
    :param model_params: tuple containing the following values:
        a_1: float. alpha1. Forgetting rate for all actions. 
        k_1: float. kappa1. Strength of reward on action value. 
        k_2: float. kappa2. Strength of omission on action value. 
    :return new_qs: Array. Updated action values. 
    '''
    a_1, k_1, k_2 = model_params
    new_qs = qs.astype('float')
    for i in [0, 1]:
        qi = new_qs[i]
        if a==i:
            if o==1:
                new_qs[i] = ((1-a_1) * qi) + (a_1*k_1)
            else:
                new_qs[i] = (1-a_1)*qi - (a_1*k_2)
        else:
            new_qs[i] = (1-a_1)*qi
    return new_qs, model_params 


def update_value_metalearning(qs, a, o, model_params):
    ''' Updates action value (q) for both possible choices according to a Q-Learning model with forgetting of unchosen action
        and parameterized uncertainty to capture expected and unexpected uncertainty on each trial. 
    :param qs: Array. Values for both choices at trial T-1. Left side value: index 0, Right side: 1. 
    :param a: integer. Side chosen on trial T. Left: 0, Right: 1. 
    :param o: integer. Outcome of trial T. Reward: 1. Nothing: 0. 
    :param model_params: tuple containing the following values:
        a_plus:       float. alpha+.  Learning rate for rewarded actions.   STATIC
        a_min:        float. alpha-.  Learning rate for unrewarded actions. DYNAMIC
        a_min_nought  float. alpha-0. Base alpha-.                          STATIC
        a_v           float. alphav.  Learning rate for RPE integration     STATIC
                                      into expected uncertainty. 
        v             float. v.       Unexpected uncertainty.               DYNAMIC
        epsilon       float. epsilon. Expected uncertainty.                 DYNAMIC
        xi            float. xi.      Forgetting rate for unchosen option.  STATIC
        psi           float. psi.     Learning rate for a_min updating.     STATIC
    :return new_qs: Array. Updated action values. 
    :return model_params: tuple containing any modifications to model parameters. 
    When fitting parameters for this model, the epsilon, nu, and a_min should be fixed at 0 when initializing.
    The fitting process takes parameters as INTITIAL values, and those should be bounded. Initial epsilon and nu can be fixed at 0. 
    They will be updated as animals engage with the environment. a_min should be fixed at a_min_nought for the initial trial.  
    Effectively, a_min, epsilon, and nu should be treated as latent variables rather than parameters. 
    They are included in the list of parameters to facilitate passing them to the next step in the iteration.

    '''


    a_plus, a_min, a_min_nought, a_v, v, epsilon, xi, psi = model_params

    new_qs = qs.astype('float')


    # Values to be used in updating Q-Values
    delta = o - new_qs[a] # Calculate prediction error
    v = abs(delta) - epsilon # Calculate unexpected uncertainty of current outcome. 

    # Update a_min
    if delta > 0:
        a_min = a_min
    else:
        a_min = psi*(v + a_min_nought) + (1-psi)*a_min
    a_min = max([0, a_min])


    # Update Q-values
    for i in [0, 1]:
        qi = new_qs[i]
        if a==i:
            # If last action was i, update based on delta:
            if delta > 0:
                new_qs[i] = qi + a_plus*delta*(1-epsilon)
            else:
                new_qs[i] = qi + a_min*delta*(1-epsilon)
        else:
            # If last action was not i, update according to forgetting rate
            new_qs[i] = xi*qi

    epsilon = epsilon + a_v*v # Calculate expected uncertainty for NEXT trial based on surprise on this trial.

    model_params = a_plus, a_min, a_min_nought, a_v, v, epsilon, xi, psi

    return new_qs, model_params


# DETERMINISTIC MODELS

def noisy_wsls(q_vals, a, o, model_params):
    ''' Generates a probability of right-sided choice based on a noisy win-stay / lose-shift given the last choice
        and outcome.
    :param epsilon: Noise parameter. 
    :param a: action on last trial.
    :param o: outcome of last trial.
    :return p_right: THe probability of making a right-sided choice given this model. 
    '''
    # right_side_win = a==1 and o == 1
    # left_side_loss = a==0 and o==0
    # right_side_loss = a==1 and o==0
    # left_side_win = a==0 and o==1
    # if right_side_win or left_side_loss:
    #     p_right = 1 - (epsilon / 2)
    # elif right_side_loss or left_side_win:
    #     p_right = epsilon/2
    # The above is an easier to read version of the below:


    epsilon, = model_params

    if a==o:
        p_right =  1- (epsilon/2)
    else:
        p_right = (epsilon/2)

    new_qs = numpy.array([1-p_right, p_right])

    return new_qs, model_params

###  CHOICE MODELS

def make_selection_SOFTMAX(option_values, beta=3):  
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


def make_selection_PROBABILISTIC(option_values, beta=None):
    '''
    A choice function for 2 options based on a simple probability.
    :param option_values: The probabilities of making left and right-sided choice. 
    :return x, p: return a choice 0 or 1, and the probability associated with choosing either option (ordered 0, 1).
    '''

    p = numpy.array(option_values)

    if random.random()<p[0]:
        return 0, p
    else:
        return 1, p
