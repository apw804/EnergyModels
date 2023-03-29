# Script to analyse network energy consumption for turning N cells off

import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
sys.path.append('/Users/apw804/dev_02/EnergyModels/KISS')

from kiss import fig_timestamp



# Set the project path
project_path = Path("~/dev_02/EnergyModels/KISS").expanduser().resolve()
project_path_str = str(project_path)
print(f'Project path:{project_path}')
data_path = project_path / 'data' / 'output' / 'switch_n_cells_off' / '2023_03_27'
s1co_data = data_path / '01_cells_off'
s2co_data = data_path / '02_cells_off'
s3co_data = data_path / '03_cells_off'
s4co_data = data_path / '04_cells_off'
s5co_data = data_path / '05_cells_off'

# Create a dataframe for each set of data by reading all the tsv files in the
#  directory and concatenating them into a dataframe per directory
for d in [s1co_data, s2co_data, s3co_data, s4co_data, s5co_data]:
    df_list = []
    for f in d.glob('*.tsv'):
        df = pd.read_csv(f, sep='\t')
        # Add a column for the experiment id
        df['experiment_id'] = f.stem
        df_list.append(df)
    # Concatenate the list of dataframes into a single dataframe
    df = pd.concat(df_list, ignore_index=True)
    # Append the serving cell id and ue id to the experiment id
    df['experiment_id'] = df['experiment_id'] + '_cell' + \
        df['serving_cell_id'].astype(str) + '_ue' + df['ue_id'].astype(str)
    # Set the index to the experiment id
    df.set_index('experiment_id', inplace=True)
    # Add the dataframe to the dictionary
    globals()[d.name] = df








# For each tsv file in the directory, read the data into a dataframe and add a
#  column for the experiment id (the file name). Then append the dataframes to
#  a list.
df_list = []
for f in rccp_data.glob('*.tsv'):
    df = pd.read_csv(f, sep='\t')
    # Add a column for the experiment id
    df['experiment_id'] = f.stem
    df_list.append(df)



# Concatenate the list of dataframes into a single dataframe
df_cc = pd.concat(df_list, ignore_index=True)

# Append the serving cell id and ue id to the experiment id
df_cc['experiment_id'] = df_cc['experiment_id'] + '_cell' + \
    df_cc['serving_cell_id'].astype(str) + '_ue' + df_cc['ue_id'].astype(str)

# Set the index to the experiment id
df_cc.set_index('experiment_id', inplace=True)