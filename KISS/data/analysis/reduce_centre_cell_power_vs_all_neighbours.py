# Script to compare cell 8 and cell 10 power

import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
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

# Convert the cell9_power column to a float
df_cc['cell9_power'] = df_cc['cell9_power'].astype(float)
# Convert the cell9_seed column to an int
df_cc['cell9_seed'] = df_cc['cell9_seed'].astype(int)

# Create a dataframe from an existing dataframe using loc to select rows
# where the serving_cell_id is 9 and another where the serving_cell_id is not 9
df_cc_only = df_cc.loc[df_cc['serving_cell_id'] == 9, :]
df_not_cc = df_cc.loc[df_cc['serving_cell_id'] != 9, :]

# Create a network dataframe
df_network = df_cc.copy()
# The number of UEs attached to each cell needs to be recorded in a new column
df_network['cell_num_ues'] = df_network.groupby(by=['cell9_seed','cell9_power', 'serving_cell_id','sc_power(dBm)'])['ue_id'].transform('count')
# We can do the same for the number of UEs in the network (should always be 400 UEs)
df_network['network_num_ues'] = df_network.groupby(by=['cell9_seed','cell9_power'])['ue_id'].transform('count')

# Group by sc_power(dBm) and seed and aggregate the mean
df_cc_condensed = df_cc_only.groupby(by=['cell9_seed','cell9_power', 'sc_power(dBm)']).agg('mean')
df_not_cc_condensed = df_not_cc.groupby(by=['cell9_seed','cell9_power', 'sc_power(dBm)']).agg('mean')

df_network = df_cc.copy()
# Count the number of UEs attached to each cell
df_network['cell_num_ues'] = df_network.groupby(by=['cell9_seed','cell9_power', 'serving_cell_id','sc_power(dBm)'])['ue_id'].transform('count')
# Count the number of UEs in the network
df_network['network_num_ues'] = df_network.groupby(by=['cell9_seed','cell9_power'])['ue_id'].transform('count')
# Rename the serving_cell_id column to cell_id
df_network = df_network.rename(columns={
    'serving_cell_id': 'cell_id',
    'sc_power(dBm)': 'cell_tx_power_dBm',
    'sc_power(watts)': 'cell_tx_power_watts',
    'cell_throughput(Mb/s)': 'cell_throughput_Mbps'
                                        })
# Drop the UE related columns and redundant rows
ue_cols = ['ue_id', 'distance_to_cell(m)', 'ue_throughput(Mb/s)', 'sinr(dB)', 'cqi', 'mcs','sc_rsrp(dBm)']
df_network = df_network.drop(ue_cols, axis=1)
df_network = df_network.drop_duplicates()

# Get the total network throughput
df_network['total_network_throughput_Mbps'] = df_network.groupby(by=['cell9_seed', 'cell9_power'])['cell_throughput_Mbps'].transform('sum')
# Get the total network power
df_network['total_network_power_kW'] = df_network.groupby(by=['cell9_seed', 'cell9_power'])['cell_power(kW)'].transform('sum')
# Get the total network energy efficiency (EE)
df_network['total_network_ee_1e-3'] = df_network['total_network_throughput_Mbps'] / df_network['total_network_power_kW']


# Get the total and mean Rest of Network (RoN) throughput
df_network['total_ron_throughput_Mbps'] = df_network.groupby(by=['cell9_seed', 'cell9_power','cell_tx_power_dBm'])['cell_throughput_Mbps'].transform('sum')
# Get the total RoN power
df_network['total_ron_power_kW'] = df_network.groupby(by=['cell9_seed', 'cell9_power','cell_tx_power_dBm'])['cell_power(kW)'].transform('sum')
# Get the total RoN energy efficiency (EE)
df_network['total_ron_ee_1e-3'] = df_network['total_ron_throughput_Mbps'] / df_network['total_ron_power_kW']

# Sanity checker: get the dataframe for a specific seed and power
# df_network_s16_c0_c9p13 = df_network.loc[:][(df_network['seed']==16) & (df_network['cell_id']==0) & (df_network['cell9_power']==13)]
# df_network_s16_c9p13 = df_network.loc[:][(df_network['seed']==16) & (df_network['cell9_power']==13)]


# Get stats for the total network throughput, power and EE
df_exp_stats = df_network.copy()
df_exp_stats = df_network.groupby(by=['cell9_power'])[['total_network_throughput_Mbps','total_network_power_kW','total_network_ee_1e-3']].agg(['mean','std'])
df_exp_stats.columns = ['_'.join(col).strip() for col in df_exp_stats.columns.values]

# Create a dataframe for the centre cell
df_cell9 = df_network.copy()
# Drop the rows where the cell_id is not 9
df_cell9 = df_cell9.loc[df_cell9['cell_id'] == 9, :]

# Get stats for the centre cell throughput, power and EE
df_cell9_stats = df_cell9.copy()
df_cell9_stats.rename(columns={'cell_ee(bits/J)':'cell_ee_1e-3'}, inplace=True)
df_cell9_stats['cell_ee_1e-3'] = df_cell9_stats['cell_ee_1e-3'] * 1e-3
df_cell9_stats = df_cell9_stats.groupby(by=['cell9_power'])[['cell_throughput_Mbps','cell_power(kW)','cell_ee_1e-3']].agg(['mean','std'])
df_cell9_stats.columns = ['_'.join(col).strip() for col in df_cell9_stats.columns.values]



# Plot the mean and standard deviation of the total network throughput, power and EE
# as a function of the cell9_power, for the whole experiment network
fig1, ax1 = plt.subplots(1, figsize=(8, 8))
# Plot the mean whole network throughput as a bar chart with error bars (std) on fig1
sns.lineplot(data=df_exp_stats, x='cell9_power', y='total_network_throughput_Mbps_mean', ax=ax1, marker='.', label='Network throughput mean')
plt.fill_between(df_exp_stats.index, df_exp_stats['total_network_throughput_Mbps_mean'] - df_exp_stats['total_network_throughput_Mbps_std'], df_exp_stats['total_network_throughput_Mbps_mean'] + df_exp_stats['total_network_throughput_Mbps_std'], alpha=0.2, label='Network throughput std')
# On the same axis, plot the mean centre cell throughput as a seaborn line plot with shaded area (std)
sns.lineplot(data=df_cell9_stats, x='cell9_power', y='cell_throughput_Mbps_mean', ax=ax1, marker='.', label='Cell 9 throughput mean')
plt.fill_between(df_cell9_stats.index, df_cell9_stats['cell_throughput_Mbps_mean'] - df_cell9_stats['cell_throughput_Mbps_std'], df_cell9_stats['cell_throughput_Mbps_mean'] + df_cell9_stats['cell_throughput_Mbps_std'], alpha=0.2, label='Cell 9 throughput std')

# Plot the network power as a line chart
fig2, ax2 = plt.subplots(1, figsize=(8, 8))
sns.lineplot(data=df_exp_stats, x='cell9_power', y='total_network_power_kW_mean', ax=ax2, marker='.', label='Network power mean')
plt.fill_between(df_exp_stats.index, df_exp_stats['total_network_power_kW_mean'] - df_exp_stats['total_network_power_kW_std'], df_exp_stats['total_network_power_kW_mean'] + df_exp_stats['total_network_power_kW_std'], alpha=0.2, label='Network power std')

# On the same axis, plot the mean centre cell power as a seaborn line plot with shaded area (std)
sns.lineplot(data=df_cell9_stats, x='cell9_power', y='cell_power(kW)_mean', ax=ax2, marker='.', label='Cell 9 power mean')
plt.fill_between(df_cell9_stats.index, df_cell9_stats['cell_power(kW)_mean'] - df_cell9_stats['cell_power(kW)_std'], df_cell9_stats['cell_power(kW)_mean'] + df_cell9_stats['cell_power(kW)_std'], alpha=0.2, label='Cell 9 power std')


# Plot the network EE as a seaborn plot with shaded area (std)
fig3, ax3 = plt.subplots(1, figsize=(8, 8))
sns.lineplot(data=df_exp_stats, x='cell9_power', y='total_network_ee_1e-3_mean', ax=ax3, marker='.', label='Network EE mean')
plt.fill_between(df_exp_stats.index, df_exp_stats['total_network_ee_1e-3_mean'] - df_exp_stats['total_network_ee_1e-3_std'], df_exp_stats['total_network_ee_1e-3_mean'] + df_exp_stats['total_network_ee_1e-3_std'], alpha=0.2, label='Network EE std')
# On the same axis, plot the centre cell EE as a seaborn line plot with shaded area (std)
sns.lineplot(data=df_cell9_stats, x='cell9_power', y='cell_ee_1e-3_mean', ax=ax3, marker='.', label='Cell 9 EE')
plt.fill_between(df_cell9_stats.index, df_cell9_stats['cell_ee_1e-3_mean'] - df_cell9_stats['cell_ee_1e-3_std'], df_cell9_stats['cell_ee_1e-3_mean'] + df_cell9_stats['cell_ee_1e-3_std'], alpha=0.2, label='Cell 9 EE std')

# Set the labels and grid
ax1.set_xlabel('Cell 9 Power (dBm)')
ax1.set_ylabel('Throughput (Mbps)')
# Set x-axis tick labels to integers
ax1.set_xticklabels(df_exp_stats.index.astype(int))
# Adjust the y-axis scale to go up in steps of 100
ax1_y_tick_steps = 100
ax1.set_yticks(np.arange(0, df_exp_stats['total_network_throughput_Mbps_mean'].max(), ax1_y_tick_steps))
# Set the x-ticks to be the same as the x-axis labels
ax1.set_xticks(df_exp_stats.index)
ax1.grid(True)
ax1.set_title('Mean Total Network and Cell9 Throughput and as a Function of Cell 9 Power')
ax1.legend()

ax2.set_xlabel('Cell 9 Power (dBm)')
ax2.set_ylabel('Power (kW)')
ax2.set_yticks(np.arange(0, df_exp_stats['total_network_power_kW_mean'].max(), 5.0))
# Set the x-ticks to be the same as the x-axis labels
ax2.set_xticks(df_exp_stats.index)
ax2.grid(True)
ax2.set_title('Mean Total Network and Cell9 Power as a Function of Cell 9 Power')
ax2.legend()


ax3.set_xlabel('Cell 9 Power (dBm)')
ax3.set_ylabel('Energy Efficiency (bits/J) * 1e-3')
ax3.set_yticks(np.arange(0, df_exp_stats['total_network_ee_1e-3_mean'].max(), 1.0))
# Set the x-ticks to be the same as the x-axis labels
ax3.set_xticks(df_exp_stats.index)
ax3.grid(True)
ax3.set_title('Mean Total Network and Cell9 EE as a Function of Cell 9 Power')
ax3.legend()
# ax3.legend(['Total Network EE mean', 'Total Network EE std', 'Cell 9 EE mean', 'Cell 9 EE std'])

# Save the figures
fig1.savefig('2023_04_05_12_02_network_and_cell9_throughput_vs_cell9_power.pdf', dpi=300, format='pdf')
fig2.savefig('2023_04_05_12_02_network_and_cell9_power_vs_cell9_power.pdf', dpi=300, format='pdf')
fig3.savefig('2023_04_05_12_02_network_and_cell9_ee_vs_cell9_power.pdf', dpi=300, format='pdf')



# Create a dataframe for the RoN
df_ron = df_network.copy()
# Drop the rows where the cell_id is 9
df_ron = df_ron.loc[df_ron['cell_id'] != 9, :]
# Group by the cell9_power, cell9_seed and cell_tx_power_dBm and drop the redundant rows
df_ron = df_ron.groupby(by=['cell9_power', 'cell9_seed','cell_tx_power_dBm']).first()
# Drop cell specific and whole network columns
df_ron = df_ron.drop(columns=['cell_id', 'cell_num_ues', 'network_num_ues', 'total_network_throughput_Mbps', 'total_network_power_kW', 'total_network_ee_1e-3', 'cell_tx_power_watts', 'cell_throughput_Mbps', 'cell_power(kW)', 'cell_ee(bits/J)', 'cell_se(bits/Hz)'])


# Create a dataframe for the centre cell
df_cell9 = df_network.copy()
# Drop the rows where the cell_id is not 9
df_cell9 = df_cell9.loc[df_cell9['cell_id'] == 9, :]
# Drop the last 7 columns
df_cell9 = df_cell9.drop(df_cell9.iloc[:, -6:], axis=1)
df_cell9 = df_cell9.drop(columns=['cell_tx_power_dBm'])

# Add cell9_seed and cell9_power columns to the index
df_cell9 = df_cell9.set_index(['cell9_seed', 'cell9_power'])
# Sort the index
df_cell9 = df_cell9.sort_index()



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


# Do a plot for centre cell power vs network throughput (Mb/s) and cell throughput (Mb/s)
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
ax2[0].errorbar(df_cc_mean.index,
                    df_cc_mean['cell_throughput(Mb/s)_mean'],
                    yerr=df_cc_mean['cell_throughput(Mb/s)_std'],
                    label='Cell 9',
                    fmt='o',
                    capsize=2,
                    linewidth=1)

ax2[0].errorbar(df_not_cc_mean.index,
                    df_not_cc_mean['cell_throughput(Mb/s)_mean'],
                    yerr=df_not_cc_mean['cell_throughput(Mb/s)_std'],
                    label='Neighbours',
                    fmt='.',
                    capsize=2,
                    linewidth=1)

ax2[0].set_xlabel('Cell 9 power output (dBm)')
ax2[0].set_ylabel('cell_throughput(Mb/s)')
ax2[0].legend()
ax2[0].grid(True)


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