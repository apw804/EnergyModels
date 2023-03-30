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

# # For each tsv file in the directory, read the data into a dataframe and add a
# #  column for the experiment id (the file name). Then append the dataframes to
# #  a list.
# df_list = []
# for f in rccp_data.glob('*.tsv'):
#     df = pd.read_csv(f, sep='\t')
#     # Add a column for the experiment id
#     df['experiment_id'] = f.stem
#     df_list.append(df)
# # Concatenate the list of dataframes into a single dataframe
# df_cc = pd.concat(df_list, ignore_index=True)
# # split the experiment_id column on the underscore and drop the last part
# df_cc["experiment_id"] = df_cc["experiment_id"].str.split("_").str[:-1].str.join("_")

# # Build a faux index from the experiment id
# # Split the experiment id on the underscore and get the 2nd part
# df_cc["cell9_seed"] = df_cc["experiment_id"].str.split("_").str[1].str.replace("s", "")
# # Split the experiment id on the underscore and take the 3rd part
# df_cc["cell9_power"] = df_cc["experiment_id"].str.split("_").str[2].str.replace("p", "")

# # write to a feather file
# df_cc.to_feather(rccp_data / '2023_03_28_rccp_data.feather')

# Read the feather file into a dataframe
df_cc = pd.read_feather(rccp_data / '2023_03_28_rccp_data.feather')

# Columns to drop
snooze_columns = ['time', 'serving_cell_sleep_mode', 'neighbour1_rsrp(dBm)', 'neighbour2_rsrp(dBm)', 'noise_power(dBm)']
df_cc = df_cc.drop(columns=snooze_columns)

# Create a dataframe from an existing dataframe using loc to select rows
# where the serving_cell_id is 9 and another where the serving_cell_id is not 9
df_cc_only = df_cc.loc[df_cc['serving_cell_id'] == 9, :]
df_not_cc = df_cc.loc[df_cc['serving_cell_id'] != 9, :]



# Group by sc_power(dBm) and seed and aggregate the mean
df_cc_condensed = df_cc_only.groupby(by=['cell9_power', 'cell9_seed','sc_power(dBm)']).agg('mean')
df_not_cc_condensed = df_not_cc.groupby(by=['cell9_power', 'cell9_seed','sc_power(dBm)']).agg('mean')

# For every seed and power(dBm) in the condensed dataframe...
interesting_cols = ['cell_throughput(Mb/s)', 
                    'cell_power(kW)', 
                    'cell_ee(bits/J)', 
                    'cell_se(bits/Hz)']

df_cc_mean = df_cc_condensed.groupby(level=['cell9_power'])[interesting_cols].agg(['mean', 'std'])
df_cc_mean.columns = ['_'.join(col).strip() for col in df_cc_mean.columns.values]

df_not_cc_mean = df_not_cc_condensed.groupby(level=['cell9_power'])[interesting_cols].agg(['mean', 'std'])
df_not_cc_mean.columns = ['_'.join(col).strip() for col in df_not_cc_mean.columns.values]



# Do a plot for centre cell power vs throughput
# Plot the cell throughput, cell power, cell ee and cell se for the centre cell,
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax[0,0].errorbar(df_cc_mean.index,
                    df_cc_mean['cell_throughput(Mb/s)_mean'],
                    yerr=df_cc_mean['cell_throughput(Mb/s)_std'],
                    label='Cell 9',
                    fmt='o', 
                    capsize=2, 
                    linewidth=1)

ax[0,0].errorbar(df_not_cc_mean.index,
                    df_not_cc_mean['cell_throughput(Mb/s)_mean'],
                    yerr=df_not_cc_mean['cell_throughput(Mb/s)_std'],
                    label='Neighbours',
                    fmt='.', 
                    capsize=2, 
                    linewidth=1)
                
ax[0, 0].set_xlabel('Cell 9 power output (dBm)')
ax[0, 0].set_ylabel('cell_throughput(Mb/s)')
ax[0, 0].legend()
ax[0, 0].grid(True)



ax[0,1].errorbar(df_cc_mean.index,
                    df_cc_mean['cell_power(kW)_mean'],
                    yerr=df_cc_mean['cell_power(kW)_std'],
                    label='Cell 9',
                    fmt='o', 
                    capsize=2, 
                    linewidth=1)
ax[0,1].errorbar(df_not_cc_mean.index,
                    df_not_cc_mean['cell_power(kW)_mean'],
                    yerr=df_not_cc_mean['cell_power(kW)_std'],
                    label='Neighbours',
                    fmt='.', 
                    capsize=2, 
                    linewidth=1)
                
ax[0, 1].set_xlabel('Cell 9 power output (dBm)')
ax[0, 1].set_ylabel('cell_power(kW)')
ax[0, 1].legend()
ax[0, 1].grid(True)



ax[1,0].errorbar(df_cc_mean.index,
                    df_cc_mean['cell_ee(bits/J)_mean'],
                    yerr=df_cc_mean['cell_ee(bits/J)_std'],
                    label='Cell 9',
                    fmt='o', 
                    capsize=2, 
                    linewidth=1)
ax[1,0].errorbar(df_not_cc_mean.index,
                    df_not_cc_mean['cell_ee(bits/J)_mean'],
                    yerr=df_not_cc_mean['cell_ee(bits/J)_std'],
                    label='Neighbours',
                    fmt='.', 
                    capsize=2, 
                    linewidth=1)
                
ax[1, 0].set_xlabel('Cell 9 power output (dBm)')
ax[1, 0].set_ylabel('cell_ee(bits/J)')
ax[1, 0].legend()
ax[1, 0].grid(True)



ax[1,1].errorbar(df_cc_mean.index,
                    df_cc_mean['cell_se(bits/Hz)_mean'],
                    yerr=df_cc_mean['cell_se(bits/Hz)_std'],
                    label='Cell 9',
                    fmt='o', 
                    capsize=2, 
                    linewidth=1)
ax[1,1].errorbar(df_not_cc_mean.index,
                    df_not_cc_mean['cell_se(bits/Hz)_mean'],
                    yerr=df_not_cc_mean['cell_se(bits/Hz)_std'],
                    label='Neighbours',
                    fmt='.', 
                    capsize=2, 
                    linewidth=1)
                
ax[1, 1].set_xlabel('Cell 9 power output (dBm)')
ax[1, 1].set_ylabel('cell_se(bits/Hz)')
ax[1, 1].legend()
ax[1, 1].grid(True)




# Dodging the datasets to avoid error bars overlapping
# df_cc_mean.index = df_cc_mean.index - 1



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
fig.suptitle('Cell 9 vs Neighbours for varying Cell 9 power levels')

# import the fig_timestamp from a file in a parent directory
fig_timestamp(fig, author='Kishan Sthankiya')
plt.tight_layout()
plt.show()
plt.savefig('/Users/apw804/dev_02/EnergyModels/KISS/2023_03_30_16_23_cell9_vs_neighbours.tiff', dpi=300, bbox_inches='tight', pad_inches=0.1)