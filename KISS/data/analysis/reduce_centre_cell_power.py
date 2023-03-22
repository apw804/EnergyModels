from sys import displayhook, path as sys_path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import subprocess

# Set the project path
project_path = Path("~/dev-02/EnergyModels/KISS").expanduser().resolve()
project_path_str = str(project_path)
print(f'Project path:{project_path}')

# Directory containing the TSV files
tsv_dir = project_path / 'data' / 'output' / 'reduce_centre_cell_power' / '2023_03_17' / 'Attempt_003'

# For each tsv file in the directory read the file into a DataFrame and append to a list
df_list = []
for tsv_file in tsv_dir.glob('*.tsv'):
    df_temp = pd.read_csv(tsv_file, sep='\t')
    df_list.append(df_temp)

# Concatenate the list of DataFrames into a single DataFrame
df = pd.concat(df_list)

# What do the first 5 rows of the DataFrame look like?
df.head()

# What does the shape look like?
df.shape

# And the info?
df.info()

# There is a LOT of noise here. I'll start with the basics for one DataFrame
# and then I'll try to figure out how to do it for all of them.

# Get the data for seed value 0 and sc_power(dBm) value 30.0
df_s000_p30 = df[(df['seed'] == 0) & (df['sc_power(dBm)'] == 30.0)]

# This SHOULD be a dataframe with 16 rows (+1 for the column names)
df_s000_p30.shape

# Let's see whatthe whole frame looks like
displayhook(df_s000_p30)

# Store the serving_cell_id and time column as a single integer since it's the same for all rows
cell_id = df_s000_p30['serving_cell_id'].iloc[0]
time = df_s000_p30['time'].iloc[0]

# Reorder by ue_id
df_s000_p30_sorted = df_s000_p30.sort_values('ue_id').copy()

"""
What we're really interested in here is the cell_throughput, cell_power, cell_ee and cell_se, which is all the same for serving_cell_id = 9 at seed = 0.

So a good summary row for serving cell 9 at seed 0 might have the columns:

seed, time, serving_cell_id, **n_ues_attached, mean_distance_to_cell, mean_ue_throughput,** sc_power, **mean_sc_rsrp, mean_sinr, mean_cqi, mean_mcs**, cell_throughput, cell_power, cell_ee, cell_se

The columns to calculate are therefore the ones above in **bold**. 
"""

# Get the number of unique ue_ids in the DataFrame
n_ues_attached = df_s000_p30_sorted['ue_id'].nunique()

print(n_ues_attached)

# Now group by serving_cell_id and get the mean for distance_to_cell, ue_throughput(Mb/s), sc_rsrp, sinr, cqi and mcs
df_s000_p30_sorted_col_means = df_s000_p30_sorted.groupby("serving_cell_id")[["distance_to_cell(m)", "sc_rsrp(dBm)", "sinr(dB)", "cqi", "mcs"]].agg('mean')

df_s000_p30_sorted_col_means.add_prefix('mean_')

# Insert the seed, time, serving_cell_id and n_ues_attached columns into the DataFrame
df_s000_p30_sorted_col_means.insert(0, 'seed', 0)
df_s000_p30_sorted_col_means.insert(1, 'time', time)
df_s000_p30_sorted_col_means.insert(2, 'cell_id', cell_id)
df_s000_p30_sorted_col_means.insert(3, 'ues_attached', n_ues_attached)

# Adding the mean_ue_throughput column is a sanity check, it should be the same as the cell_throughput column
df_s000_p30_sorted_col_means.insert(6, 'mean_ue_throughput(Mb/s)', df_s000_p30_sorted['ue_throughput(Mb/s)'].mean())

# Add the cell_throughput, cell_power, cell_ee, cell_se columns which are the same for all rows
df_s000_p30_sorted_col_means.insert(5, 'cell_throughput(Mb/s)', df_s000_p30_sorted['cell_throughput(Mb/s)'].iloc[0])
df_s000_p30_sorted_col_means.insert(4, 'cell_power(dBm)', df_s000_p30_sorted['sc_power(dBm)'].iloc[0])
df_s000_p30_sorted_col_means.insert(7, 'cell_ee', df_s000_p30_sorted['cell_ee(bits/J)'].iloc[0])
df_s000_p30_sorted_col_means.insert(8, 'cell_se', df_s000_p30_sorted['cell_se(bits/Hz)'].iloc[0])


displayhook(df_s000_p30_sorted_col_means)

print(df_s000_p30_sorted_col_means.columns)