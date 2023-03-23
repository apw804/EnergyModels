# Script to analyse the data from the cell 5 power test

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


# Set the project path
project_path = Path("~/dev-02/EnergyModels/KISS").expanduser().resolve()
project_path_str = str(project_path)
print(f'Project path:{project_path}')

# Read in the CSV file
df = pd.read_csv(project_path / 'data' / 'output' / 'reduce_cell_5_power' / '2023_03_17' / 'reduce_cell_5_power.csv')

# Sort by the seed and sc_power(dBm) columns
df_sorted = df.sort_values(['seed', 'sc_power(dBm)'])

# Drop the 'Unamed: 0', time and serving_cell_sleep_mode columns
df_sorted.drop(columns=['Unnamed: 0', 'time', 'serving_cell_sleep_mode'], inplace=True)

# Move the sc_power(dBm) column to the right of the serving_cell_id column
sc_power_col = df_sorted.pop('sc_power(dBm)')
df_sorted.insert(2, 'sc_power(dBm)', sc_power_col)

# What do the first 5 rows of the DataFrame look like?
print(df_sorted.head())

# Count the number of unique ue_ids when grouped by seed and serving_cell_id
print(df_sorted.groupby(['seed', 'serving_cell_id'])['ue_id'].nunique())

# Add this as a column to the DataFrame and call it n_ues_attached
df_sorted['n_ues_attached'] = df_sorted.groupby(['seed', 'serving_cell_id'])['ue_id'].transform('nunique')

# Move the n_ues_attached column to the right of the sc_power(dBm) column
n_ues_attached_col = df_sorted.pop('n_ues_attached')
df_sorted.insert(3, 'n_ues_attached', n_ues_attached_col)

# What do the first 5 rows of the DataFrame look like?
print(df_sorted.head())

# Group by seed and serving_cell_id and sc_power(dBm) sorted as ascending, ascending, descending - and calculate the mean of the remaining columns
df_grouped = df_sorted.groupby(['seed', 'serving_cell_id', 'sc_power(dBm)']).mean().sort_values(['seed', 'serving_cell_id', 'sc_power(dBm)'], ascending=[True, True, True])

# Drop any rows where the serving_cell_id index is not 5
df_grouped = df_grouped.drop(df_grouped[df_grouped.index.get_level_values('serving_cell_id') != 5].index)

# Remove the serving_cell_id and seed index levels and reinsert them as columns
df_grouped = df_grouped.reset_index(level=['serving_cell_id', 'seed'])

# Sort by the sc_power(dBm) and then the seed columns
df_grouped = df_grouped.sort_values(['sc_power(dBm)', 'seed'])

# column labels
new_columns = ['cell_id', 'cell_output_dBm', 'total_seeds', 'energy_cons_mean(kW)', 
               'energy_cons_std(kW)', 'ee_mean(bits/J)', 'ee_std(bits/J)', 
               'se_mean(bits/J)', 'se_std(bits/J)', 'n_ues_mean', 'n_ues_std']

# Construct a new Dataframe with the index as the first column
df_cell_5_power = pd.DataFrame(df_grouped['sc_power(dBm)'])

# Add serving_cell_id as 'cell_id' column
df_cell_5_power['cell_id'] = df_grouped['serving_cell_id']

# For each sc_power(dBm) value, count the number of unique seeds and add this as a column
df_cell_5_power['total_seeds'] = df_grouped.groupby(['sc_power(dBm)'])['seed'].transform('nunique')


# What do the first 5 rows of the DataFrame look like?
print(df_cell_5_power.head())

# What does the shape look like?
print(df_cell_5_power.shape)

# How many unique ue_ids are there?
print(df_cell_5_power['cell_id'].nunique())




# # Different approach. Read in one TSV to a dataframe and do the analysis.
# # Set the directory containing the TSV files
# data_path = project_path / 'data' / 'output' / 'reduce_cell_5_power' / '2023_03_17' / 'reduce_cell_5_power'

# # Make a list of the TSV files
# tsv_list = list(data_path.glob('*.tsv'))

# # Read the first TSV file into a DataFrame
# df = pd.read_csv(tsv_list[0], sep='\t')

# # What do the first 5 rows of the DataFrame look like?
# print(df.head())

# # What does the shape look like?
# print(df.shape)

# # How many unique ue_ids are there?
# print(df['ue_id'].nunique())



