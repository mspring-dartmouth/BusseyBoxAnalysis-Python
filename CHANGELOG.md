#### VERSION 0.1a3 ####
	- Altered method by which busboxanalysis.freeforaging.grid_search() scores parameters. 
	  The previous method, which used numpy.prod, was overflowing for certain inputs, erroneously returning 0.0 instead of explicitly raising an error. The new method calculates the sum of logarithms applied to the list of choice probabilities, which ought to be robust to longer lists of small values. Also added docstrings to functions lacking them in previous versions, including busboxanalysis.freeforaging.grid_search().


#### VERSION 0.1a2 ####
	- Developing module for 5CSRTT analysis.
	- Adding base functions as necessary to complement the above. 
	- Bug fixes from 0.1a1

#### VERSION 0.1a1 ####
	- Initial commit. Included a set of base functions and specialized functions for freeforaging and reinforcement learning analysis. 