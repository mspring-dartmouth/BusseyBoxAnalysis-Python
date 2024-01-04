import pandas as pd
import natsort
from busboxanalysis import get_final_values


# Turn this into a function to summarize responses into each window for a given animal on a given day. 

def summarize_responses(raw_input_dataframe):
	'''
		DOCSTRING
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

