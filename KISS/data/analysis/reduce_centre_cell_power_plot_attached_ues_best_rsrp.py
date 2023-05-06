# Script to plot the number of UE's attached to each cell for different levels of cell 9 power

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
data_path = project_path / 'data' / 'output' / 'reduce_centre_cell_power' / '2023_04_12'
rccp_data = data_path / 'rsrp_cell_to_zero_watts'

# Create a generator function that will yeild a dataframe from the tsv files in a directory
def tsv_to_df_generator(dir_path):
    for f in dir_path.glob('*.tsv'):
        df = pd.read_csv(f, sep='\t')
        # Add a column for the experiment id
        df['experiment_id'] = f.stem
        # Split the experiment_id column on the underscore and drop the last part
        df["experiment_id"] = df["experiment_id"].str.split("_").str[:-1].str.join("_")
        # Split the experiment id on the underscore and get the 2nd part
        df["cell9_seed"] = df["experiment_id"].str.split("_").str[1].str.replace("s", "")
        # Split the experiment id on the underscore and take the 3rd part
        df["cell9_power"] = df["experiment_id"].str.split("_").str[2].str.replace("p", "")
        yield df

# Create a place to store the data
cell_ue_count_array_list = []

# Create a generator object from the generator function
df_generator = tsv_to_df_generator(rccp_data)

# Loop through the generator
for df in df_generator:
    # Capture the cell9 power and seed for this experiment
    cell9_power = df['cell9_power'].unique()[0]
    if cell9_power == '':
        cell9_power = -2
    cell9_seed = df['cell9_seed'].unique()[0]

    # Get the number of UEs attached to each cell
    df_cell_ue_count = df.groupby('serving_cell_id')['ue_id'].nunique().sort_index() # FIXME - may need to set a zero value for cell9
    # Transform the Series into an array
    df_cell_ue_count = df_cell_ue_count.to_numpy()
    # Add the cell9 power and seed to the beginning of the array
    df_cell_ue_count = np.insert(df_cell_ue_count, 0, cell9_power)
    df_cell_ue_count = np.insert(df_cell_ue_count, 1, cell9_seed)

    # Add the array to the list
    cell_ue_count_array_list.append(df_cell_ue_count)

# Create a dataframe from the list of arrays
df_cell_ue_count = pd.DataFrame(cell_ue_count_array_list)

# Create a list of column names from the number of cells
column_names = ['cell9_power', 'cell9_seed']
for i in range(19):
    column_names.append(f'cell{i}_ue_count')

# Set the column names
df_cell_ue_count.columns = column_names

# Group the dataframe by cell9 power and get the mean and standard deviation
df_cell_ue_count = df_cell_ue_count.groupby('cell9_power').agg(['mean', 'std'])

# Create a figure to plot the mean number of UEs attached to cells for different Cell 9 Power Levels
fig, ax = plt.subplots(figsize=(8, 6))

# Lists of neighbours
immediate_neighbours = [4, 5, 8, 10, 13, 14]
distant_neighbours = [0, 1, 2, 3, 6, 7, 11, 12, 15, 16, 17, 18]


# Plot the mean number of UEs attached to immediate neighbour cells for different Cell 9 Power Levels
for i in immediate_neighbours:
    df_cell_ue_count[f'cell{i}_ue_count']['mean'].plot(ax=ax, marker='.', label=f'Cell {i}')

# Plot the mean number of UEs attached to distant neighbour cells for different Cell 9 Power Levels
for i in distant_neighbours:
    df_cell_ue_count[f'cell{i}_ue_count']['mean'].plot(ax=ax, marker='x', label=f'Cell {i}')

# Plot the mean number of UEs attached to Cell 9 for different Cell 9 Power Levels
df_cell_ue_count['cell9_ue_count']['mean'].plot(ax=ax, marker='o', label='Cell 9')


# Add a legend and position it to the right of the plot
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# Add a title
ax.set_title('Mean Number of UEs Attached to Cells for Different Cell 9 Power Levels down to Zero Watts\n with Best RSRP handover strategy')

# Add axis labels
ax.set_xlabel('Cell 9 Power Level (dBm)')
ax.set_ylabel('Mean Number of UEs Attached to Cells')

# Set the x axis ticks to Cell9 power levels
ax.set_xticks(df_cell_ue_count.index)

# Set the x axis tick label `0` to -inf
ax.set_xticklabels([f'{x}' if x != -2 else '-inf' for x in df_cell_ue_count.index])

# Add grid lines with dashes
ax.grid(linestyle='--')

# Save the figure
fig.savefig(project_path / 'data' / 'figures'  / '2023_04_12_mean_ues_attached_to_cells_for_different_cell_9_power_levels_to_zero_watts_best_rsrp.pdf', format='pdf')