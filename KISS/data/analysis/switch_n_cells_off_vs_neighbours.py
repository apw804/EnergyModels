# Script to look at the effect of reducing the power of the centre cell on the
#  throughput, power, energy efficiency and spectral efficiency of the centre
#  cell and its immediate neighbours.

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

# snco = [s1co_data, s2co_data, s3co_data, s4co_data, s5co_data]

# # For each folder and each tsv file in the directory, read the data into a dataframe and add a
# #  column for the experiment id (the file name). Then append the dataframes to
# #  a list.
# df_list = []
# for i in snco:
#   df_sublist = []
#   for f in i.glob('*.tsv'):
#       df = pd.read_csv(f, sep='\t')
#       # Add a column for the experiment id
#       df['experiment_id'] = f.stem
#       df_sublist.append(df)
#   # Concatenate the list of dataframes into a single sub-dataframe
#   df_sub_snco = pd.concat(df_sublist, ignore_index=True)

#   # Append the sub-dataframe to the list of dataframes
#   df_list.append(df_sub_snco)

# # Concatenate the list of dataframes into a single dataframe
# df_snco = pd.concat(df_list, ignore_index=True)

# # Build a faux index from the experiment id
# # Get the second character (a number) from the experiment_id string
# df_snco["num_cells_off"] = df_snco["experiment_id"].str[1]
# # Split the experiment id on the underscore and get the 2nd part
# df_snco["snco_seed"] = df_snco["experiment_id"].str.split("_").str[1].str.replace("s", "")
# # Split the experiment id on the underscore and take the 3rd part
# df_snco["snco_power"] = df_snco["experiment_id"].str.split("_").str[2].str.replace("p", "")

# # write to a feather file
# df_snco.to_feather(data_path / '2023_03_27_snco_data.feather')

# Read the feather file into a dataframe
df_snco = pd.read_feather(data_path / '2023_03_27_snco_data.feather')

# Columns to drop
snooze_columns = ['time', 'serving_cell_sleep_mode', 'neighbour1_rsrp(dBm)', 'neighbour2_rsrp(dBm)', 'noise_power(dBm)']
df_snco = df_snco.drop(columns=snooze_columns)


# Should've logged the cells that were switched off, but didn't. So, we'll
#  have to do it the hard way. 

# For every experiment_id, get the list of cells that were switched off
#  and add a column to the dataframe with an ordered list of the cells that
#  were switched off.

# Get the list of experiment ids
experiment_ids = df_snco['experiment_id'].unique()
# Create a list to hold the dataframes
df_list = []
# For each experiment id...
for e in experiment_ids:
    # Create a dataframe from an existing dataframe using loc to select rows
    #  where the experiment_id is equal to the current experiment id
    df_e = df_snco.loc[df_snco['experiment_id'] == e, :].copy()

    # Get the list of cells that were switched off
    cells_off = df_e.loc[df_e['sc_power(dBm)'] == 0.0, 'serving_cell_id'].unique()

    # Add a column to the dataframe with an ordered list of the cells that
    #  were switched off and fill the column with the list of cells that were
    #  switched off
    cells_off_col = np.array2string(cells_off, separator=',')
    df_e['cells_off'] = cells_off_col

    # Append the dataframe to the list of dataframes
    df_list.append(df_e)

# Concatenate the list of dataframes into a single dataframe
df_better_snco = pd.concat(df_list, ignore_index=True)

# Group by num_cells_off, cell_off, snco_seed and snco_power and aggregate the mean for the interesting columns
interesting_cols = ['cell_throughput(Mb/s)', 
                    'cell_power(kW)', 
                    'cell_ee(bits/J)', 
                    'cell_se(bits/Hz)']

df_better_snco_grouped = df_better_snco.groupby(['num_cells_off', 'cells_off', 'snco_seed', 'snco_power'])[interesting_cols].agg(['mean', 'std']).copy()
df_better_snco_grouped.columns = ['_'.join(col).strip() for col in df_better_snco_grouped.columns.values]




df_snco_mean = df_snco_condensed.groupby(level=['cell9_power'])[interesting_cols].agg(['mean', 'std'])
df_snco_mean.columns = ['_'.join(col).strip() for col in df_snco_mean.columns.values]

df_in_mean = df_in_condensed.groupby(level=['cell9_power'])[interesting_cols].agg(['mean', 'std'])
df_in_mean.columns = ['_'.join(col).strip() for col in df_in_mean.columns.values]



# Do a plot for centre cell power vs throughput
# Plot the cell throughput, cell power, cell ee and cell se for the centre cell,
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax[0,0].errorbar(df_snco_mean.index,
                    df_snco_mean['cell_throughput(Mb/s)_mean'],
                    yerr=df_snco_mean['cell_throughput(Mb/s)_std'],
                    label='Cell 9',
                    fmt='o', 
                    capsize=2, 
                    linewidth=1)

ax[0,0].errorbar(df_in_mean.index,
                    df_in_mean['cell_throughput(Mb/s)_mean'],
                    yerr=df_in_mean['cell_throughput(Mb/s)_std'],
                    label='Immediate Neighbours',
                    fmt='.', 
                    capsize=2, 
                    linewidth=1)
                
ax[0, 0].set_xlabel('Cell 9 power output (dBm)')
ax[0, 0].set_ylabel('cell_throughput(Mb/s)')
ax[0, 0].legend()
ax[0, 0].grid(True)



ax[0,1].errorbar(df_snco_mean.index,
                    df_snco_mean['cell_power(kW)_mean'],
                    yerr=df_snco_mean['cell_power(kW)_std'],
                    label='Cell 9',
                    fmt='o', 
                    capsize=2, 
                    linewidth=1)
ax[0,1].errorbar(df_in_mean.index,
                    df_in_mean['cell_power(kW)_mean'],
                    yerr=df_in_mean['cell_power(kW)_std'],
                    label='Immediate Neighbours',
                    fmt='.', 
                    capsize=2, 
                    linewidth=1)
                
ax[0, 1].set_xlabel('Cell 9 power output (dBm)')
ax[0, 1].set_ylabel('cell_power(kW)')
ax[0, 1].legend()
ax[0, 1].grid(True)



ax[1,0].errorbar(df_snco_mean.index,
                    df_snco_mean['cell_ee(bits/J)_mean'],
                    yerr=df_snco_mean['cell_ee(bits/J)_std'],
                    label='Cell 9',
                    fmt='o', 
                    capsize=2, 
                    linewidth=1)
ax[1,0].errorbar(df_in_mean.index,
                    df_in_mean['cell_ee(bits/J)_mean'],
                    yerr=df_in_mean['cell_ee(bits/J)_std'],
                    label='Immediate Neighbours',
                    fmt='.', 
                    capsize=2, 
                    linewidth=1)
                
ax[1, 0].set_xlabel('Cell 9 power output (dBm)')
ax[1, 0].set_ylabel('cell_ee(bits/J)')
ax[1, 0].legend()
ax[1, 0].grid(True)



ax[1,1].errorbar(df_snco_mean.index,
                    df_snco_mean['cell_se(bits/Hz)_mean'],
                    yerr=df_snco_mean['cell_se(bits/Hz)_std'],
                    label='Cell 9',
                    fmt='o', 
                    capsize=2, 
                    linewidth=1)
ax[1,1].errorbar(df_in_mean.index,
                    df_in_mean['cell_se(bits/Hz)_mean'],
                    yerr=df_in_mean['cell_se(bits/Hz)_std'],
                    label='Immediate Neighbours',
                    fmt='.', 
                    capsize=2, 
                    linewidth=1)
                
ax[1, 1].set_xlabel('Cell 9 power output (dBm)')
ax[1, 1].set_ylabel('cell_se(bits/Hz)')
ax[1, 1].legend()
ax[1, 1].grid(True)


# Dodging the datasets to avoid error bars overlapping
# df_snco_mean.index = df_snco_mean.index - 1

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
fig.suptitle('Cell 9 vs Immediate Neighbours for varying Cell 9 power levels')

# import the fig_timestamp from a file in a parent directory
fig_timestamp(fig, author='Kishan Sthankiya')
plt.tight_layout()
plt.savefig('cell9_vs_immediate_neighbours.png', dpi=300)