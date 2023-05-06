import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
sys.path.append('/Users/apw804/dev_02/EnergyModels/KISS')
import seaborn as sns
from kiss import fig_timestamp


# Set the project path
project_path = Path("~/dev_02/EnergyModels/KISS").expanduser().resolve()
project_path_str = str(project_path)
print(f'Project path:{project_path}')
data_path = project_path / 'data' / 'output' / 'reduce_Y_cell_5_and_8_and_14_power' / '2023_05_03'
rcp_data = data_path
cells_turned_down = []

# Create a generator function that will yeild a dataframe from the tsv files in a directory
def tsv_to_df_generator(dir_path):
    for f in dir_path.glob('*.tsv'):
        df = pd.read_csv(f, sep='\t')
        # Add a column for the experiment id
        df['experiment_id'] = f.stem
        # Split the experiment_id column on the underscore and drop the last part
        df["experiment_id"] = df["experiment_id"].str.split("_").str[:-1].str.join("_")
        # Split the experiment id on the underscore and get the 2nd part
        df["reduced_cells_seed"] = df["experiment_id"].str.split("_").str[1].str.replace("s", "")

        # Detect the cells that are turned down from the dir_path
        rcp_cells = []
        # Get the sub-string between 'reduce_cell_' and '_power' in the dir_path
        string = str(f)
        sub_string = string[string.find("reduce_cell_") + len("reduce_cell_"):string.rfind("_power")]
        # Split the sub-string on the underscore
        sub_string = sub_string.split("_")
        # Get strings that are digits and convert to int
        for cell in sub_string:
            if cell.isdigit():
                rcp_cells.append(int(cell))


        # Add columns for each of the rcp_cells where the value is the sc_power(dBm) for that cell, or -np.inf if indicated by the experiment_id
        for cell in rcp_cells:
            if 'inf' in df['experiment_id'].values[0]:
                # Set the reduced_cells_{cell}_power_dBm to -inf
                df[f'reduced_cell_{cell}_power'] = -np.inf
            
            df[f'reduced_cell_{cell}_power'] = df.loc[df['serving_cell_id'] == cell, 'sc_power(dBm)'].values[0]

            # Convert all values in the reduced_cells_{cell}_power_dBm column to floats
            df[f'reduced_cell_{cell}_power'] = df[f'reduced_cell_{cell}_power'].astype(float)

        yield df, rcp_cells

# Create a generator object from the generator function
df_generator = tsv_to_df_generator(rcp_data)

# Create a list to store the dataframes
df_list = []

# Loop through the generator
for df, rcp_cells in df_generator:

    # If the rcp_cells do not match the cells_turned_down, then update the cells_turned_down
    if rcp_cells != cells_turned_down:
        cells_turned_down = rcp_cells

    # Get the number of UEs per serving_cell_id, by counting the number of unique ue_id's where the sc_power_watts is greater than 0.0
    # Group the DataFrame by 'serving_cell_id', and count the number of non-zero 'sc_power(watts)' values
    result = df.loc[df['sc_power(watts)'] > 0].groupby('serving_cell_id')['sc_power(watts)'].count()

    # create a Series with zero counts for any 'serving_cell_id' values that did not have non-zero 'sc_power(watts)' values
    zero_counts = pd.Series(0, index=df['serving_cell_id'].unique())
    ue_count = result.reindex(zero_counts.index, fill_value=0)

    # Calculate the average of all columns per serving_cell_id, by dividing the sum by the number of UE's per serving_cell_id
    df_sc_average = df.copy()
    # Check the data type in each column and sum only the numeric columns but retain the non-numeric columns, and ignore the seed and time columns
    df_sc_average = df_sc_average.groupby('serving_cell_id').sum(numeric_only=True).div(ue_count, axis=0)

    # Add the non-numeric columns back to the dataframe
    df_sc_average = df_sc_average.join(df.groupby('serving_cell_id').first()[['experiment_id', 'reduced_cells_seed']])

    # Replace the ue_id column with the number of UEs per serving_cell_id and rename the column to ue_count
    df_sc_average['ue_id'] = ue_count
    df_sc_average = df_sc_average.rename(columns={'ue_id': 'ue_count'})
    df_sc_average.reset_index(inplace=True)

    # Add the dataframe to the list
    df_list.append(df_sc_average)

    # Find dataframes where the serving_cell_id is 8 and the correspoding seed value is inf


# Concatenate the dataframes in the list into a single dataframe
df_concat = pd.concat(df_list)

# View the dataframe
print(df_concat.head())

# View the datatypes in the dataframe
print(df_concat.dtypes)

# Convert the reduced_cell_x_power columns to int
for cell in cells_turned_down:
    df_concat[f'reduced_cell_{cell}_power'] = df_concat[f'reduced_cell_{cell}_power'].astype(float)


# At this stage the dataframe has 16 rows for each combination of:
# - serving_cell_id
# - seed
#
# This is because there are 15 different power levels for cells that are turned down +1 for the OFF
# state of the reduced cells.

# For each cells_turned_down, we need to add another row for the cell SLEEP state.
# This will be the same as the OFF state, but with the serving_cell_sleep_mode set to 3 and the sc_power(watts) set to 780.0
for cell in cells_turned_down:
    df_concat.reset_index(inplace=False)
    # Get unique seed values
    unique_seeds = df_concat['reduced_cells_seed'].unique()
    # For each seed value add a row with the below values
    for seed in unique_seeds:
        if pd.isna(seed):
            seed = -1e9
        # Add the cell OFF state row
        df_concat = df_concat.append({
            'serving_cell_id': cell, 
            'seed': int(seed), 
            'time': 1,
            'serving_cell_sleep_mode': 0, 
            'ue_count': 0, 
            'distance_to_cell(m)': 0.0, 
            'sc_power(watts)': 0.0, 
            'ue_throughput(Mb/s)': 0.0,
            'sc_power(dBm)': -np.inf,
            'sc_power(watts)': 0.0,
            'sc_rsrp(dBm)': -np.inf,
            'neighbour1_rsrp(dBm)': -np.inf,
            'neighbour2_rsrp(dBm)': -np.inf,
            'noise_power(dBm)': -np.inf,
            'sinr(dB)': -np.inf,
            'cqi': 0, 
            'mcs': 0, 
            'cell_throughput(Mb/s)': 0.0,
            'cell_power(kW)': 0.00,
            'cell_ee(bits/J)': 0.0,
            'cell_se(bits/Hz)': 0.0,
            'reduce_cells_seed': int(seed),
            f'reduced_cell_{cell}_power': 0.0,
            'experiment_id': 'tweaked_OFF',
            # For the reduced_cell_x_power columns, where x is NOT the cell that is turned down, set the value to -2
            f'reduced_cell_{[x for x in cells_turned_down if x != cell][0]}_power': 0.0,
            }, ignore_index=True)
        # Add the cell SLEEP state row
        df_concat = df_concat.append({
            'serving_cell_id': cell, 
            'seed': int(seed), 
            'time': 1,
            'serving_cell_sleep_mode': 4,
            'ue_count': 0, 
            'distance_to_cell(m)': 0.0, 
            'sc_power(watts)': 0.0, 
            'ue_throughput(Mb/s)': 0.0,
            'sc_power(dBm)': -999,      # Pretending this means a sleep state
            'sc_power(watts)': 0.0,
            'sc_rsrp(dBm)': -np.inf,
            'neighbour1_rsrp(dBm)': -np.inf,
            'neighbour2_rsrp(dBm)': -np.inf,
            'noise_power(dBm)': -np.inf,
            'sinr(dB)': -np.inf,
            'cqi': 0, 
            'mcs': 0, 
            'cell_throughput(Mb/s)': 0.0,
            'cell_power(kW)': 0.780,
            'cell_ee(bits/J)': 0.0,
            'cell_se(bits/Hz)': 0.0,
            'reduce_cells_seed': int(seed),
            f'reduced_cell_{cell}_power': -999,
            'experiment_id': 'tweaked_SLEEP',
            f'reduced_cell_{[x for x in cells_turned_down if x != cell][0]}_power': 0.0,
            }, ignore_index=True)
        
# Print the types of data in the reduced_cell_x_power columns
print(df_concat[[f'reduced_cell_{cell}_power' for cell in cells_turned_down]].dtypes)

# Drop level_0 and index columns
# df_concat.drop(columns=['level_0', 'index'], inplace=True)

# Group by reduced_cell_5_power (and cell 8 and cell 14) for the whole network
df_stats = df_concat.groupby(['sc_power(dBm)']).agg({
    'cell_throughput(Mb/s)': ['mean', 'std'],
    'cell_power(kW)': ['mean', 'std'],
    'cell_ee(bits/J)': ['mean', 'std'],
    'cell_se(bits/Hz)': ['mean', 'std']
    })
df_stats.columns = ['_'.join(col).strip() for col in df_stats.columns.values]

# Get rows for just the inner_ring_cells
df_inner_ring_cells = df_concat[df_concat['serving_cell_id'].isin(rcp_cells)]

# Get all other rows
df_other_cells = df_concat[~df_concat['serving_cell_id'].isin(rcp_cells)]

# Group by reduced_cell_5_power (and cell 8 and cell 14) AND serving_cell_id for the inner ring cells
df_inner_ring_stats = df_inner_ring_cells.groupby(['reduced_cell_5_power', 'serving_cell_id']).agg({
    'cell_throughput(Mb/s)': ['mean', 'std'],
    'cell_power(kW)': ['mean', 'std'],
    'cell_ee(bits/J)': ['mean', 'std'],
    'cell_se(bits/Hz)': ['mean', 'std']
    })
df_inner_ring_stats.columns = ['_'.join(col).strip() for col in df_inner_ring_stats.columns.values]

# Group by reduced_cell_5_power (and cell 8 and cell 14) AND serving_cell_id for the other cells
df_other_cells_stats = df_other_cells.groupby(['reduced_cell_5_power', 'serving_cell_id']).agg({
    'cell_throughput(Mb/s)': ['mean', 'std'],
    'cell_power(kW)': ['mean', 'std'],
    'cell_ee(bits/J)': ['mean', 'std'],
    'cell_se(bits/Hz)': ['mean', 'std']
    })
df_other_cells_stats.columns = ['_'.join(col).strip() for col in df_other_cells_stats.columns.values]

# Reset the index for the dataframes
df_stats.reset_index(inplace=True)
df_inner_ring_stats.reset_index(inplace=True)
df_other_cells_stats.reset_index(inplace=True)

# Reframe the data for plotting
# Get the mean and standard deviaton of the inner ring cells and other cells per reduced_cell_5_power
df_reduced_power_cells = df_inner_ring_stats.groupby(['reduced_cell_5_power']).agg({
    'cell_throughput(Mb/s)_mean': ['mean', 'std'],
    'cell_power(kW)_mean': ['mean', 'std'],
    'cell_ee(bits/J)_mean': ['mean', 'std'],
    'cell_se(bits/Hz)_mean': ['mean', 'std']
    })
df_reduced_power_cells.columns = ['_'.join(col).strip() for col in df_reduced_power_cells.columns.values]

#### FIXME - You were removing the repeated _mean_mean and mean_std from the column names, but this is not needed

# Get the mean and standard deviaton of the inner ring cells and other cells per reduced_cell_5_power
df_full_power_cells = df_other_cells_stats.groupby(['reduced_cell_5_power']).agg({
    'cell_throughput(Mb/s)_mean': ['mean', 'std'],
    'cell_power(kW)_mean': ['mean', 'std'],
    'cell_ee(bits/J)_mean': ['mean', 'std'],
    'cell_se(bits/Hz)_mean': ['mean', 'std']
    })
df_full_power_cells.columns = ['_'.join(col).strip() for col in df_full_power_cells.columns.values]


# Reset the index for the dataframes
df_reduced_power_cells.reset_index(inplace=True)
df_full_power_cells.reset_index(inplace=True)



# Plot the cell throughput, cell power, cell ee and cell se for the centre cell,
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# Set figure title
fig.suptitle(f'Effects of reducing the output power (watts) of cell 5, cell 8 and cell 14 on the network', fontsize=16)

# Convert `reduced_cell_5_power` from dBm to watts
df_reduced_power_cells['reduced_cell_5_power_watts'] = 10**(df_reduced_power_cells['reduced_cell_5_power']/10)/1000
df_full_power_cells['reduced_cell_5_power_watts'] = 10**(df_full_power_cells['reduced_cell_5_power']/10)/1000

# Plot the cell_throughput(Mb/s) mean and standard deviation for each serving_cell_id for the df_reduced_power_cells AND df_full_power_cells with standard deviation error bars
ax[0,0].errorbar(df_reduced_power_cells['reduced_cell_5_power_watts'],
            df_reduced_power_cells['cell_throughput(Mb/s)_mean_mean'],
            yerr=df_reduced_power_cells['cell_throughput(Mb/s)_mean_std'],
            fmt='.',
            capsize=2,
            linewidth=1,
            label='Reduced Tx Power Cells Mean (non-adjacent inner ring)',
)

ax[0,0].errorbar(df_full_power_cells['reduced_cell_5_power_watts'],
            df_full_power_cells['cell_throughput(Mb/s)_mean_mean'],
            yerr=df_full_power_cells['cell_throughput(Mb/s)_mean_std'],
            fmt='.',
            capsize=2,
            linewidth=1,
            label='Full Tx Power Cells Mean')
ax[0,0].set_xlim(0, 20)

# Set the x and y axis labels
ax[0,0].set_xlabel('Reduced Cell Tx Power (watts)')
ax[0,0].set_ylabel('Mean Cell Throughput (Mb/s)')
ax[0,0].legend()
ax[0,0].grid(True)
ax[0,0].set_title('Cell Throughput vs Reduced Tx Cell (5 & 8 & 14) Power')


# Drop the value at `reduce_cell_8_power` of 0
df_reduced_power_cells = df_reduced_power_cells[df_reduced_power_cells['reduced_cell_5_power'] != 0]
df_full_power_cells = df_full_power_cells[df_full_power_cells['reduced_cell_5_power'] != 0]


# Cell power(kW)
# `reduced_cell_5_power_watts` with -999 value is not plotted
ax[0,1].errorbar(df_reduced_power_cells[df_reduced_power_cells['reduced_cell_5_power'] != -999]['reduced_cell_5_power_watts'],
            df_reduced_power_cells[df_reduced_power_cells['reduced_cell_5_power'] != -999]['cell_power(kW)_mean_mean'],
            yerr=df_reduced_power_cells[df_reduced_power_cells['reduced_cell_5_power'] != -999]['cell_power(kW)_mean_std'],
            fmt='o',
            capsize=2,
            linewidth=1,
            label='Reduced Power Cell Mean',
            )

# Plot the cell SLEEP power (with clipping turned off so that the point is visible)
sleep_power_watts = df_reduced_power_cells[df_reduced_power_cells['reduced_cell_5_power'] == -999]['reduced_cell_5_power_watts']
ax[0,1].plot(0,
            df_reduced_power_cells[df_reduced_power_cells['reduced_cell_5_power'] == -999]['cell_power(kW)_mean_mean'],
            marker='o',
            markersize=5,
            color='red',
            label='Reduced Power Cell Sleep Power',
            markeredgecolor='#1f77b4', 
            markeredgewidth=3,
            clip_on=False,
            )

ax[0,1].errorbar(df_full_power_cells['reduced_cell_5_power_watts'],
            df_full_power_cells['cell_power(kW)_mean_mean'],
            yerr=df_full_power_cells['cell_power(kW)_mean_std'],
            fmt='.',
            capsize=2,
            linewidth=1,
            label='Full Power Cell Mean',
            )
ax[0,1].set_xlim(0, 20)

# Set the x and y axis labels
ax[0,1].set_xlabel('Reduced Cell Tx Power (watts)')
ax[0,1].set_ylabel('Mean Cell Power (kW)')
ax[0,1].legend()
ax[0,1].grid(True)
ax[0,1].set_title('Cell Power vs Reduced Tx Cell (5 & 8 & 14) Power')



ax[1,0].errorbar(df_reduced_power_cells['reduced_cell_5_power_watts'],
            df_reduced_power_cells['cell_ee(bits/J)_mean_mean'],
            yerr=df_reduced_power_cells['cell_ee(bits/J)_mean_std'],
            fmt='.',
            capsize=2,
            linewidth=1,
            label='Reduced Tx Power Cells',
            )

ax[1,0].errorbar(df_full_power_cells['reduced_cell_5_power_watts'],
            df_full_power_cells['cell_ee(bits/J)_mean_mean'],
            yerr=df_full_power_cells['cell_ee(bits/J)_mean_std'],
            fmt='.',
            capsize=2,
            linewidth=1,
            label='Full Tx Power Cells',
            )
ax[1,0].set_xlim(0, 20)

# Set the x and y axis labels
ax[1,0].set_xlabel('Reduced Cell Tx Power (watts)')
ax[1,0].set_ylabel('Mean Cell EE (bits/J)')
ax[1,0].legend()
ax[1,0].grid(True)
ax[1,0].set_title('Cell EE vs Reduced Tx Cell (5 & 8 & 14) Power')

ax[1,1].errorbar(df_reduced_power_cells['reduced_cell_5_power_watts'],
            df_reduced_power_cells['cell_se(bits/Hz)_mean_mean'],
            yerr=df_reduced_power_cells['cell_se(bits/Hz)_mean_std'],
            fmt='.',
            capsize=2,
            linewidth=1,
            label='Reduced Tx Power Cells',
            )

ax[1,1].errorbar(df_full_power_cells['reduced_cell_5_power_watts'],
            df_full_power_cells['cell_se(bits/Hz)_mean_mean'],
            yerr=df_full_power_cells['cell_se(bits/Hz)_mean_std'],
            fmt='.',
            capsize=2,
            linewidth=1,
            label='Full Tx Power Cells',
            )
ax[1,1].set_xlim(0, 20)

# Set the x and y axis labels
ax[1,1].set_xlabel('Reduced Cell Tx Power (watts)')
ax[1,1].set_ylabel('Mean Cell SE (bits/Hz)')
ax[1,1].legend()
ax[1,1].grid(True)
ax[1,1].set_title('Cell SE vs Reduced Tx Cell (5 & 8 & 14) Power')

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

# Set all axes subplots y axis lower limit to 0
ax[0, 0].set_ylim(bottom=0)
ax[0, 1].set_ylim(bottom=0)
ax[1, 0].set_ylim(bottom=0)
ax[1, 1].set_ylim(bottom=0)

# Set all axes subplots x axis lower limit to 0
ax[0, 0].set_xlim(left=0)
ax[0, 1].set_xlim(left=0)
ax[1, 0].set_xlim(left=0)
ax[1, 1].set_xlim(left=0)

# Auto adjust the padding between subplot
fig.tight_layout(pad=3.0)

# Save the figure
fig.savefig('2023_05_03T_r3Ycp_cell_vs_reduced_tx_cell_5_and_8_and_14_power_watts.pdf', dpi=300, format='pdf', bbox_inches='tight')

