# BusseyBoxAnalysis-Python
========
 This project will work toward analyzing mouse behavior in various experimental groups across days, as well as in varying paradigms. 

 A suite of core tools will be used to read in raw data files as DataFrames and extract data from those Frames. Specialized modules will be used for analyses of individual behavioral paradigms (e.g. 5CSRTT, dynamic foraging, etc.). 

 This package is currently in beta, and functionality is still being actively updated.

# HOW TO USE:
The first step in analyzing data for any behavior is to read in raw data using either read_raw_file() or batch_read_files() from the core package. The latter can also be used to read data from mice across multiple days. 

With the data read into Python, functions from any submodules can be used to analyze target variables for specified behaviors. These functions are generally pretty well commented (I hope), so look into the code to determine
whether any existing functions meet your purpose. All analysis is based on identifying behavioral events from a combination of evnt_ids and item_names and pulling timestamps associated with those events from the raw data. 

A basic example of using this code to read in data for a batch of animals that completed the Five-Choice Serial Reaction Time Task follows:

```python
import glob
import busboxanalysis as bba
import natsort
import pandas as pd

from datetime import datetime
def sort_date(str_date):
    dt_obj = datetime.strptime(str_date, '%m/%d/%Y')
    return dt_obj.date()

behavior_directory = 'Path/to/Files'
behavior_files = glob.glob(os.path.join(behavior_directory, '*ProtocolName.csv'))

mice = bba.batch_read_files(behavior_files) # Creates dictionary of Key: Date, Value: {Key: MouseID, Value: Data}
# Note that in the above call, the option to use the data as a Key in dictionary creation is a default. 
# This option can be turned off by setting use_date_as_key = False during the function call. 

# I then wanted to create a dataframe summarizing Accuracy, Omission %, and Total Trials over training for all animals

combo_dex = pd.MultiIndex.from_product([np.arange(nDaysOfTraining), bba.fivechoice.basemetrics])
summary_df = pd.DataFrame(index = combo_dex, columns = sorted(mice.keys()))

for mouse in mice:
    mouse_series = bba.concatenate_values_over_days(animal_dictionary=mice[mouse],
                                                    dates=natsort.natsorted(mouse_days[mouse], key= sort_date),
                                                    outer_metrics=bba.fivechoice.basemetrics)
    summary_df.loc[(range(len(mouse_days[mouse])), bba.five_choice_basemetrics, mouse] = mouse_series.values
```

And that's the fundamental process for any behavior. The precise implementation will vary with the behavior, as each of the submodules was custom written for the particular Bussey Box script used to run the behavior. 
