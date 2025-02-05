#### VERSION 0.4b  ####
	- Created separate file to contain models for rl module. Also, began standardizing argument format across models to facilitate comparison across models and simplify addition of new models. 
#### VERSION 0.3b  ####
	- Implemented a suite of hidden Markov model tools in rl module. 
#### VERSION 0.1b  ####
	- Autoshaping module is undergoing testing to reanalyze older data. 
#### VERSION 0.1a5 ####
	- Created module for analyzing autoshaping data from Bussey Boxes. As yet, it is uncommented and not fully tested. 
#### VERSION 0.1a4 ####
	- Complete overhaul of rl module. All current functionality is not contained within a Class object. The included model is now based on Q-learning with differential forgetting (https://doi.org/10.1523/jneurosci.6157-08.2009) and a soft-max decision function with variable temperature (this latter part has remained the same). The best-fit parameters are meant to be determined in conjuction with scipy.optimize.minimum (I may make this internal to the class at some point, but it is not yet), rather than by a grid search based method. Once I am comfortable with this implementation, I will remove the previous functions that are outside of the RL_mod class. 
#### VERSION 0.1a3 ####
	- Altered method by which busboxanalysis.freeforaging.grid_search() scores parameters. 
	  The previous method, which used numpy.prod, was overflowing for certain inputs, erroneously returning 0.0 instead of explicitly raising an error. The new method calculates the sum of logarithms applied to the list of choice probabilities, which ought to be robust to longer lists of small values. Also added docstrings to functions lacking them in previous versions, including busboxanalysis.freeforaging.grid_search().


#### VERSION 0.1a2 ####
	- Developing module for 5CSRTT analysis.
	- Adding base functions as necessary to complement the above. 
	- Bug fixes from 0.1a1

#### VERSION 0.1a1 ####
	- Initial commit. Included a set of base functions and specialized functions for freeforaging and reinforcement learning analysis. 