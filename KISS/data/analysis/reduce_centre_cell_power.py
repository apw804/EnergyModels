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
data_path = project_path / 'data' / 'output' / 'reduce_centre_cell_power' / '2023_03_28'
rccp_data = data_path / 'reduce_centre_cell_power'

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

# Sort the dataframes by the sc_power(dBm) and seed columns
df_cc.sort_values(by=['sc_power(dBm)', 'seed'], inplace=True)

# Create a dataframe from an existing dataframe using loc to select rows
# where the serving_cell_id is 9 and another where the serving_cell_id is not 9
# and a copy of the original dataframe
df_cc_only = df_cc.loc[df_cc['serving_cell_id'] == 9, :]
df_not_cc = df_cc.loc[df_cc['serving_cell_id'] != 9, :]


# Drop time, serving_cell_sleep_mode, neighbour1_rsrp(dBm),
# neighbour2_rsrp(dBm) and noise_power(dBm)
df_cc_only = df_cc_only.drop(columns=['time', 
                                  'serving_cell_sleep_mode',
                                  'neighbour1_rsrp(dBm)', 
                                  'neighbour2_rsrp(dBm)',
                                  'noise_power(dBm)'])
df_not_cc = df_not_cc.drop(columns=['time',
                                 'serving_cell_sleep_mode',
                                    'neighbour1_rsrp(dBm)',
                                    'neighbour2_rsrp(dBm)',
                                    'noise_power(dBm)'])


# Sort the dataframes by the sc_power(dBm) and seed columns
df_cc_only.sort_values(by=['sc_power(dBm)', 'seed'], inplace=True)
df_not_cc.sort_values(by=['sc_power(dBm)', 'seed'], inplace=True)


# Bring the sc_power(dBm) column to the far left
sc_power_cc = df_cc_only.pop('sc_power(dBm)')
df_cc_only.insert(0, 'sc_power(dBm)', sc_power_cc)

sc_power_not_cc = df_not_cc.pop('sc_power(dBm)')
df_not_cc.insert(0, 'sc_power(dBm)', sc_power_not_cc)




# Add sc_power(dBm), seed and ue_id columns to the index in ascending order
df_cc_only.set_index(['sc_power(dBm)', 'seed'], inplace=True)
df_cc_only.sort_index(inplace=True)

df_not_cc.set_index(['sc_power(dBm)', 'seed'], inplace=True)
df_not_cc.sort_index(inplace=True)



# Keep the last 8 columns and drop the rest
df_cc_only = df_cc_only.iloc[:, -8:]
df_not_cc = df_not_cc.iloc[:, -8:]



# Group by sc_power(dBm) and seed and aggregate the first value
df_cc_condensed = df_cc_only.groupby(level=['sc_power(dBm)', 'seed']).agg({
    'cell_throughput(Mb/s)': 'first',
    'cell_power(kW)': 'first',
'cell_ee(bits/J)': 'first',
'cell_se(bits/Hz)': 'first'
})
df_not_cc_condensed = df_not_cc.groupby(level=['sc_power(dBm)', 'seed']).agg({
    'cell_throughput(Mb/s)': 'first',
    'cell_power(kW)': 'first',
'cell_ee(bits/J)': 'first',
'cell_se(bits/Hz)': 'first'
})


# Group by sc_power(dBm) and aggregate the mean and standard deviation of
# the cell_throughput(Mb/s), cell_power(kW), cell_ee(bits/J) and 
# cell_se(bits/Hz) columns and rename the columns
df_cc_mean = df_cc_condensed.groupby(level='sc_power(dBm)').agg({
'cell_throughput(Mb/s)': ['mean', 'std'],
'cell_power(kW)': ['mean', 'std'],
'cell_ee(bits/J)': ['mean', 'std'],
'cell_se(bits/Hz)': ['mean', 'std']
})
df_cc_mean.columns = ['_'.join(col).strip() for col in df_cc_mean.columns.values]

df_not_cc_mean = df_not_cc_condensed.groupby(level='sc_power(dBm)').agg({
'cell_throughput(Mb/s)': ['mean', 'std'],
'cell_power(kW)': ['mean', 'std'],
'cell_ee(bits/J)': ['mean', 'std'],
'cell_se(bits/Hz)': ['mean', 'std']
})
df_not_cc_mean.columns = ['_'.join(col).strip() for col in df_not_cc_mean.columns.values]


# Dodging the datasets to avoid error bars overlapping
df_cc_mean.index = df_cc_mean.index - 1


# Plot the cell throughput, cell power, cell ee and cell se for the centre cell,
# not_centre_cell and network vs sc_power(dBm) with error bars representing the 
# standard deviation of  the mean with a line cap size of 2 and a line width of
# 1 and a grid on the plot.
fig, ax = plt.subplots(2, 2, figsize=(6, 6))
ax[0, 0].errorbar(df_cc_mean.index, 
                  df_cc_mean['cell_throughput(Mb/s)_mean'], 
                  yerr=df_cc_mean['cell_throughput(Mb/s)_std'], 
                  fmt='o', 
                  label='Cell 9', 
                  capsize=2, 
                  linewidth=1)

ax[0, 0].errorbar(df_not_cc_mean.index,
                    df_not_cc_mean['cell_throughput(Mb/s)_mean'],
                    yerr=df_not_cc_mean['cell_throughput(Mb/s)_std'],
                    fmt='x',
                    label='Not Cell 9',
                    capsize=2,
                    linewidth=1)

ax[0, 0].set_title('Cell Throughput')
ax[0, 0].set_xlabel('SC Power (dBm)')
ax[0, 0].set_ylabel('Throughput (Mb/s)')



# Set grid on the plot with a line style of '--' and a line width of 0.5
ax[0, 0].grid(linestyle='--', linewidth=0.5)


# Add minor ticks to the plot
ax[0, 0].minorticks_on()


# Set the title for the figure
fig.suptitle('Cell 9 vs. Not Cell 9')

# import the fig_timestamp from a file in a parent directory
fig_timestamp(fig, author='Kishan Sthankiya')
plt.tight_layout()
plt.show()
