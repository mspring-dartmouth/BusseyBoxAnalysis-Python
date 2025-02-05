from ._rl import (RL_mod,
				  calc_norm_likelihood,
				  generate_transition_matrix,
				  generate_markov_data,
				  HMM_Fit)

from ._models import (update_value_prediction_error,
					  update_value_dfq_static,
					  update_value_fq_static,
					  update_value_metalearning,
					  noisy_wsls,
					  make_selection_SOFTMAX,
					  make_selection_PROBABILISTIC)