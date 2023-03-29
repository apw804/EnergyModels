# Script to compare cell 8 and cell 10 power

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
data_path = project_path / 'data' / 'output' / 'reduce_cell_8_vs_10_power' / '2023_03_28'
rcp8_data = data_path / 'reduce_cell_8_power'
rcp10_data = data_path / 'reduce_cell_10_power'

# For each tsv file in the directory, read the data into a dataframe and add a
#  column for the experiment id (the file name). Then append the dataframes to
#  a list.
df8_list = []
for f in rcp8_data.glob('*.tsv'):
    df = pd.read_csv(f, sep='\t')
    # Add a column for the experiment id
    df['experiment_id'] = f.stem
    df8_list.append(df)

df10_list = []
for f in rcp10_data.glob('*.tsv'):
    df = pd.read_csv(f, sep='\t')
    # Add a column for the experiment id
    df['experiment_id'] = f.stem
    df10_list.append(df)

# Concatenate the list of dataframes into a single dataframe
df8 = pd.concat(df8_list, ignore_index=True)
df10 = pd.concat(df10_list, ignore_index=True)

# Append the serving cell id and ue id to the experiment id
df8['experiment_id'] = df8['experiment_id'] + '_cell' + \
    df8['serving_cell_id'].astype(str) + '_ue' + df8['ue_id'].astype(str)
df10['experiment_id'] = df10['experiment_id'] + '_cell' + \
    df10['serving_cell_id'].astype(str) + '_ue' + df10['ue_id'].astype(str)

# Set the index to the experiment id
df8.set_index('experiment_id', inplace=True)
df10.set_index('experiment_id', inplace=True)

# Sort the dataframes by the sc_power(dBm) and seed columns
df8.sort_values(by=['sc_power(dBm)', 'seed'], inplace=True)
df10.sort_values(by=['sc_power(dBm)', 'seed'], inplace=True)

# Create a dataframe from an existing dataframe using loc to select rows
# where the serving_cell_id is 8 and 10 respectively and assign it to a new
# variable name (df8_only and df10_only)
df8_only = df8.loc[df8['serving_cell_id'] == 8, :]
df10_only = df10.loc[df10['serving_cell_id'] == 10, :]

# Drop time, serving_cell_sleep_mode, neighbour1_rsrp(dBm),
# neighbour2_rsrp(dBm) and noise_power(dBm)
df8_only = df8_only.drop(columns=['time', 
                                  'serving_cell_sleep_mode',
                                  'neighbour1_rsrp(dBm)', 
                                  'neighbour2_rsrp(dBm)',
                                  'noise_power(dBm)'])
df10_only = df10_only.drop(columns=['time', 
                                    'serving_cell_sleep_mode',
                                    'neighbour1_rsrp(dBm)', 
                                    'neighbour2_rsrp(dBm)',
                                    'noise_power(dBm)'])

# Sort the dataframes by the sc_power(dBm) and seed columns
df8_only.sort_values(by=['sc_power(dBm)', 'seed'], inplace=True)
df10_only.sort_values(by=['sc_power(dBm)', 'seed'], inplace=True)

# Bring the sc_power(dBm) column to the far left
sc_power8 = df8_only.pop('sc_power(dBm)')
df8_only.insert(0, 'sc_power(dBm)', sc_power8)
sc_power10 = df10_only.pop('sc_power(dBm)')
df10_only.insert(0, 'sc_power(dBm)', sc_power10)

# Add sc_power(dBm), seed and ue_id columns to the index in ascending order
df8_only.set_index(['sc_power(dBm)', 'seed'], inplace=True)
df8_only.sort_index(inplace=True)

df10_only.set_index(['sc_power(dBm)', 'seed'], inplace=True)
df10_only.sort_index(inplace=True)

# Keep the last 8 columns and drop the rest
df8_only = df8_only.iloc[:, -8:]
df10_only = df10_only.iloc[:, -8:]

# Group by sc_power(dBm) and seed and aggregate the first value
df8_condensed = df8_only.groupby(level=['sc_power(dBm)', 'seed']).agg({
    'cell_throughput(Mb/s)': 'first',
    'cell_power(kW)': 'first',
'cell_ee(bits/J)': 'first',
'cell_se(bits/Hz)': 'first'
})
df10_condensed = df10_only.groupby(level=['sc_power(dBm)', 'seed']).agg({
'cell_throughput(Mb/s)': 'first',
'cell_power(kW)': 'first',
'cell_ee(bits/J)': 'first',
'cell_se(bits/Hz)': 'first'
})

# Group by sc_power(dBm) and aggregate the mean and standard deviation of
# the cell_throughput(Mb/s), cell_power(kW), cell_ee(bits/J) and 
# cell_se(bits/Hz) columns and rename the columns
df8_mean = df8_condensed.groupby(level='sc_power(dBm)').agg({
'cell_throughput(Mb/s)': ['mean', 'std'],
'cell_power(kW)': ['mean', 'std'],
'cell_ee(bits/J)': ['mean', 'std'],
'cell_se(bits/Hz)': ['mean', 'std']
})
df8_mean.columns = ['_'.join(col).strip() for col in df8_mean.columns.values]

df10_mean = df10_condensed.groupby(level='sc_power(dBm)').agg({
'cell_throughput(Mb/s)': ['mean', 'std'],
'cell_power(kW)': ['mean', 'std'],
'cell_ee(bits/J)': ['mean', 'std'],
'cell_se(bits/Hz)': ['mean', 'std']
})
df10_mean.columns = ['_'.join(col).strip() for col in df10_mean.columns.values]


# Plot the cell throughput, cell power, cell ee and cell se for cell 8 and cell
# 10 vs sc_power(dBm) with error bars representing the standard deviation of 
# the mean with a line cap size of 2 and a line width of 1 and a grid on the 
# plot.
fig, ax = plt.subplots(2, 2, figsize=(6, 6))
ax[0, 0].errorbar(df8_mean.index, 
                  df8_mean['cell_throughput(Mb/s)_mean'], 
                  yerr=df8_mean['cell_throughput(Mb/s)_std'], 
                  fmt='o', 
                  label='Cell 8', 
                  capsize=2, 
                  linewidth=1)
ax[0, 0].errorbar(df10_mean.index, 
                  df10_mean['cell_throughput(Mb/s)_mean'], 
                  yerr=df10_mean['cell_throughput(Mb/s)_std'], 
                  fmt='.', 
                  label='Cell 10', 
                  capsize=2, 
                  linewidth=1)
ax[0, 0].set_xlabel('sc_power(dBm)')
ax[0, 0].set_ylabel('Cell Throughput (Mb/s)')
ax[0, 0].legend()

ax[0, 1].errorbar(df8_mean.index, 
                  df8_mean['cell_power(kW)_mean'], 
                  yerr=df8_mean['cell_power(kW)_std'], 
                  fmt='o', 
                  label='Cell 8', 
                  capsize=2,
                  linewidth=1)
ax[0, 1].errorbar(df10_mean.index, 
                  df10_mean['cell_power(kW)_mean'], 
                  yerr=df10_mean['cell_power(kW)_std'], 
                  fmt='.', 
                  label='Cell 10', 
                  capsize=2, 
                  linewidth=1)
ax[0, 1].set_xlabel('sc_power(dBm)')
ax[0, 1].set_ylabel('Cell Power (kW)')
ax[0, 1].legend()

ax[1, 0].errorbar(df8_mean.index, 
                  df8_mean['cell_ee(bits/J)_mean'], 
                  yerr=df8_mean['cell_ee(bits/J)_std'], 
                  fmt='o', 
                  label='Cell 8', 
                  capsize=2, 
                  linewidth=1)
ax[1, 0].errorbar(df10_mean.index, 
                  df10_mean['cell_ee(bits/J)_mean'], 
                  yerr=df10_mean['cell_ee(bits/J)_std'], 
                  fmt='.', 
                  label='Cell 10', 
                  capsize=2, 
                  linewidth=1)
ax[1, 0].set_xlabel('sc_power(dBm)')
ax[1, 0].set_ylabel('Cell EE (bits/J)')
ax[1, 0].legend()

ax[1, 1].errorbar(df8_mean.index, 
                  df8_mean['cell_se(bits/Hz)_mean'], 
                  yerr=df8_mean['cell_se(bits/Hz)_std'], 
                  fmt='o', 
                  label='Cell 8', 
                  capsize=2, 
                  linewidth=1)
ax[1, 1].errorbar(df10_mean.index, 
                  df10_mean['cell_se(bits/Hz)_mean'], 
                  yerr=df10_mean['cell_se(bits/Hz)_std'], 
                  fmt='.', 
                  label='Cell 10', 
                  capsize=2, 
                  linewidth=1)
ax[1, 1].set_xlabel('sc_power(dBm)')
ax[1, 1].set_ylabel('Cell SE (bits/Hz)')
ax[1, 1].legend()

# Set grid on the plot with a line style of '--' and a line width of 0.5
ax[0, 0].grid(linestyle='--', linewidth=0.5)
ax[0, 1].grid(linestyle='--', linewidth=0.5)
ax[1, 0].grid(linestyle='--', linewidth=0.5)
ax[1, 1].grid(linestyle='--', linewidth=0.5)

# Add minor ticks to the plot
ax[0, 0].minorticks_on()
ax[0, 1].minorticks_on()
ax[1, 0].minorticks_on()
ax[1, 1].minorticks_on()

# Set the title for the figure
fig.suptitle('Cell 8 and Cell 10 vs sc_power(dBm) with error bars representing the standard deviation of the mean')

# import the fig_timestamp from a file in a parent directory
fig_timestamp(fig, author='Kishan Sthankiya')
plt.tight_layout()
plt.show()
