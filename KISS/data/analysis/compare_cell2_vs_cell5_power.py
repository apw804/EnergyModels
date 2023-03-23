# Script to analyse the data from the cell 5 power test

import datetime
from sys import displayhook
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


# Set the project path
project_path = Path("~/dev-02/EnergyModels/KISS").expanduser().resolve()
project_path_str = str(project_path)
print(f'Project path:{project_path}')


# Read in the cell2 CSV file
df2 = pd.read_csv(project_path / 'data' / 'output' / 'reduce_cell_2_power' / '2023_03_17' / 'reduce_cell_2_power.csv')

# Read in the cell5 CSV file
df5 = pd.read_csv(project_path / 'data' / 'output' / 'reduce_cell_5_power' / '2023_03_17' / 'reduce_cell_5_power.csv')

# # Create a new dataframe that copies the cell2 dataframe with columns serving_cell_id, sc_power(dBm), seed, cell_power(kW), cell_ee(bits/J), cell_se(bits/Hz)
# df2_v_df5 = df2[["serving_cell_id", "sc_power(dBm)", "seed", "cell_power(kW)", "cell_ee(bits/J)", "cell_se(bits/Hz)"]].copy()

# # Add the suffix _df2 to the columns
# df2_v_df5.columns = df2_v_df5.columns + "_df2"

# # Join the cell5 dataframe to the cell2 dataframe
# df2_v_df5 = df2_v_df5.join(df5[["serving_cell_id", "sc_power(dBm)", "seed", "cell_power(kW)", "cell_ee(bits/J)", "cell_se(bits/Hz)"]].set_index(["serving_cell_id", "sc_power(dBm)", "seed"]), on=["serving_cell_id", "sc_power(dBm)", "seed"], rsuffix="_df5")



# # Write the dataframe to a CSV file
# df2_v_df5.to_csv(project_path / 'data' / 'output' / 'compare_cell_2_vs_cell_5_power.csv', index=False)


def set_data_path(data_dir_str:str, project_path: Path,):
    """ 
    Gets the data path relative to the project path.
    """

    # Split the data path string into a list
    data_dir_list = data_dir_str.split("/")

    # The data_path is a Path object with project path plus the data path list. It should be a directory
    data_dir = project_path / Path(*data_dir_list)

    while not data_dir.exists():
        file_name = data_dir.stem
        data_dir = data_dir.parent
    for child in data_dir.iterdir():
        if callable(child):
          if child.is_file():
              if file_name in child.stem:
                file_name = child.name
                data_path = child.parent
                break
    else:
        file_name = None
        print("File not found")
        data_path = data_dir
    print(f"Data path: {data_path}")
    print(f"File name: {file_name}")
    return data_path, file_name

def filter_power_consumption_data(df: pd.DataFrame, serving_cell_id: int):
    """
    Filters out rows where the serving cell is equal to the provided serving_cell_id.
    Gets the rows where the power consumption is unique.
    Keeps the seed, sc_power(dBm), and cell_power(kW) columns.
    Returns rows with unique combinations of sc_power and cell_power.
    Converts the cell_power(kW) column to watts.
    Renames the columns.
    Sorts the dataframe by the sc_power column.
    Stores the P_out(dBm) and P_cons(W) columns as a list of tuples.
    Converts the P_out(dBm) column to watts.
    Stores the P_out(W) and P_cons(W) columns as a list of tuples.

    Args:
    df (pandas.DataFrame): A dataframe containing columns named "serving_cell_id", "cell_power(kW)", "sc_power(dBm)",
    and "seed".
    serving_cell_id (int): The ID of the serving cell to filter the data for.

    Returns:
    AIMM_sim_model_dBm:
      List of tuples containing the power consumption (W) and power output data, measured in dBm, for unique combinations of sc_power and cell_power.
    AIMM_sim_model_watts:
        List of tuples containing the power consumption (W) and power output data, measured in watts, for unique combinations of sc_power and cell_power.
    """

    df_serving_cell = df[df["serving_cell_id"] == serving_cell_id]
    df_serving_cell = df_serving_cell[df_serving_cell["cell_power(kW)"].duplicated(keep=False)]
    df_output_vs_cons = df_serving_cell[["seed", "sc_power(dBm)", "cell_power(kW)"]]
    df_output_vs_cons = df_output_vs_cons.drop_duplicates(subset=["sc_power(dBm)", "cell_power(kW)"])
    df_output_vs_cons["cell_power(W)"] = df_output_vs_cons["cell_power(kW)"] * 1e3
    df_output_vs_cons = df_output_vs_cons.rename(columns={"sc_power(dBm)": "P_out(dBm)", "cell_power(W)": "P_cons(W)"})
    df_output_vs_cons = df_output_vs_cons.sort_values(by="P_out(dBm)")
    AIMM_sim_model_dBm = list(zip(df_output_vs_cons["P_out(dBm)"], df_output_vs_cons["P_cons(W)"]))
    df_output_vs_cons["P_out(W)"] = 10 ** (df_output_vs_cons["P_out(dBm)"] / 10.0) / 1000.0
    AIMM_sim_model_watts = list(zip(df_output_vs_cons["P_out(W)"], df_output_vs_cons["P_cons(W)"]))
    return AIMM_sim_model_dBm, AIMM_sim_model_watts

# Filter the data for the serving cell with ID 2
cell2_power_dBm, cell2_power_watts = filter_power_consumption_data(df2, 2)

# Filter the data for the serving cell with ID 5
cell5_power_dBm, cell5_power_watts = filter_power_consumption_data(df5, 5)

# Create a dataframe from the cell2_power_dBm list of tuples
df_cell2_power_dBm = pd.DataFrame(cell2_power_dBm, columns=["P_out(dBm)", "P_cons(W)"])

# Create a dataframe from the cell5_power_dBm list of tuples
df_cell5_power_dBm = pd.DataFrame(cell5_power_dBm, columns=["P_out(dBm)", "P_cons(W)"])

# Join the two dataframes
df_cell2_vs_cell5_power_dBm = df_cell2_power_dBm.join(df_cell5_power_dBm, lsuffix="_cell2", rsuffix="_cell5")

# Show the dataframe
print(df_cell2_vs_cell5_power_dBm)

# Compare the power consumption for the two serving cells (dBm)
fig1, ax1 = plt.subplots()
ax1.plot(*zip(*cell2_power_dBm), label="Cell 2", marker="o")
ax1.plot(*zip(*cell5_power_dBm), label="Cell 5", marker="x", linestyle="--")
plt.xlabel("Power output (dBm)")
plt.ylabel("Power consumption (W)")
plt.title("Power consumption vs power output (dBm) for cell 2 and cell 5")
plt.grid()
plt.legend()
plt.show()

# Compare the power consumption for the two serving cells (watts)
fig2, ax2 = plt.subplots()
ax2.plot(*zip(*cell2_power_watts), label="Cell 2", marker=".")
ax2.plot(*zip(*cell5_power_watts), label="Cell 5", marker="x", linestyle="--")
plt.xlabel("Power output (W)")
plt.ylabel("Power consumption (W)")
plt.title("Power consumption vs power output (W) for cell 2 and cell 5")
plt.grid()
plt.legend()
plt.show()


data_path, file_name = set_data_path(data_dir_str='data/analysis', project_path=project_path)

# Save the figures to disk in the project folder with today's date and timestamp
figure_path = data_path.parent / "figures"
figure_path.mkdir(parents=True, exist_ok=True)
today = datetime.datetime.today().strftime("%Y_%m_%d")
now = datetime.datetime.now().strftime("%H_%M_%S")
fig1.savefig(f"{figure_path}/{today}_{now}_compare_cell2_vs_cell5_dBm.png", dpi=300)
fig2.savefig(f"{figure_path}/{today}_{now}_compare_cell2_vs_cell5_watts.png", dpi=300)

# # Sort by the seed and sc_power(dBm) columns
# df_sorted = df.sort_values(['seed', 'sc_power(dBm)'])

# # Drop the 'Unamed: 0', time and serving_cell_sleep_mode columns
# df_sorted.drop(columns=['Unnamed: 0', 'time', 'serving_cell_sleep_mode'], inplace=True)

# # Move the sc_power(dBm) column to the right of the serving_cell_id column
# sc_power_col = df_sorted.pop('sc_power(dBm)')
# df_sorted.insert(2, 'sc_power(dBm)', sc_power_col)

# # What do the first 5 rows of the DataFrame look like?
# print(df_sorted.head())

# # Count the number of unique ue_ids when grouped by seed and serving_cell_id
# print(df_sorted.groupby(['seed', 'serving_cell_id'])['ue_id'].nunique())

# # Add this as a column to the DataFrame and call it n_ues_attached
# df_sorted['n_ues_attached'] = df_sorted.groupby(['seed', 'serving_cell_id'])['ue_id'].transform('nunique')

# # Move the n_ues_attached column to the right of the sc_power(dBm) column
# n_ues_attached_col = df_sorted.pop('n_ues_attached')
# df_sorted.insert(3, 'n_ues_attached', n_ues_attached_col)

# # What do the first 5 rows of the DataFrame look like?
# print(df_sorted.head())

# # Group by seed and serving_cell_id and sc_power(dBm) sorted as ascending, ascending, descending - and calculate the mean of the remaining columns
# df_grouped = df_sorted.groupby(['seed', 'serving_cell_id', 'sc_power(dBm)']).mean().sort_values(['seed', 'serving_cell_id', 'sc_power(dBm)'], ascending=[True, True, True])

# # Drop any rows where the serving_cell_id index is not 5
# df_grouped = df_grouped.drop(df_grouped[df_grouped.index.get_level_values('serving_cell_id') != 5].index)

# # Remove the serving_cell_id and seed index levels and reinsert them as columns
# df_grouped = df_grouped.reset_index(level=['serving_cell_id', 'seed'])

# # Sort by the sc_power(dBm) and then the seed columns
# df_grouped = df_grouped.sort_values(['sc_power(dBm)', 'seed'])

# # column labels
# new_columns = ['cell_id', 'cell_output_dBm', 'total_seeds', 'energy_cons_mean(kW)', 
#                'energy_cons_std(kW)', 'ee_mean(bits/J)', 'ee_std(bits/J)', 
#                'se_mean(bits/J)', 'se_std(bits/J)', 'n_ues_mean', 'n_ues_std']

# # Construct a new Dataframe with the index as the first column
# df_cell_5_power = pd.DataFrame(df_grouped['sc_power(dBm)'])

# # Add serving_cell_id as 'cell_id' column
# df_cell_5_power['cell_id'] = df_grouped['serving_cell_id']

# # For each sc_power(dBm) value, count the number of unique seeds and add this as a column
# df_cell_5_power['total_seeds'] = df_grouped.groupby(['sc_power(dBm)'])['seed'].transform('nunique')


# # What do the first 5 rows of the DataFrame look like?
# print(df_cell_5_power.head())

# # What does the shape look like?
# print(df_cell_5_power.shape)

# # How many unique ue_ids are there?
# print(df_cell_5_power['cell_id'].nunique())







