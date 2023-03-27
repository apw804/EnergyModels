# Script to analyse the data from the cell 5 power test

import datetime
from sys import displayhook
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


# Set the project path
project_path = Path("~/dev_02/EnergyModels/KISS").expanduser().resolve()
project_path_str = str(project_path)
print(f'Project path:{project_path}')
rcp2_data = project_path / 'data' / 'output' / 'reduce_cell_2_power' / '2023_03_17' / 'reduce_cell_2_power'
rcp5_data = project_path / 'data' / 'output' / 'reduce_cell_5_power' / '2023_03_17' / 'reduce_cell_5_power'


# Read in the cell2 CSV file
df2 = pd.read_csv( 'reduce_cell_2_power.csv', index_col=False)

# Read in the cell5 CSV file
df5 = pd.read_csv(project_path / 'data' / 'output' / 'reduce_cell_5_power' / '2023_03_17' / 'reduce_cell_5_power.csv', index_col=False)

# Sort the dataframes by the sc_power(dBm) and seed columns
df2 = df2.sort_values(by=["sc_power(dBm)", "seed"], ascending=True)
df5 = df5.sort_values(by=["sc_power(dBm)", "seed"], ascending=True)

# Drop the Unnamed: 0 column from both dataframes
df2 = df2.drop(columns=["Unnamed: 0"])
df5 = df5.drop(columns=["Unnamed: 0"])

# Drop index, serving_cell_sleep_mode and noise_power(dBm) columns from both dataframes
df2 = df2.drop(columns=["serving_cell_sleep_mode", "noise_power(dBm)"])
df5 = df5.drop(columns=["serving_cell_sleep_mode", "noise_power(dBm)"])

# Drop the index
df2 = df2.reset_index(drop=True)
df5 = df5.reset_index(drop=True)

# Filter rows where the serving_cell_id is 2
df2 = df2[df2["serving_cell_id"] == 2]

# Filter rows where the serving_cell_id is 5
df5 = df5[df5["serving_cell_id"] == 5]



# # Create a new dataframe that copies the cell2 dataframe columns serving_cell_id, sc_power(dBm), seed, cell_power(kW), cell_ee(bits/J), cell_se(bits/Hz), ue_id, distance_to_cell(m).
# df2_v_df5 = df2[["serving_cell_id", "sc_power(dBm)", "seed", "cell_power(kW)", "cell_ee(bits/J)", "cell_se(bits/Hz)", "ue_id", "ue_throughput(Mb/s)", "distance_to_cell(m)"]]

# # Join the cell5 dataframe to the df2_v_df5 dataframe columns serving_cell_id, sc_power(dBm), seed, cell_power(kW), cell_ee(bits/J), cell_se(bits/Hz), ue_id, distance_to_cell(m).
# df2_v_df5 = df2_v_df5.join(df5[["serving_cell_id", "sc_power(dBm)", "seed", "cell_power(kW)", "cell_ee(bits/J)", "cell_se(bits/Hz)", "ue_id", "ue_throughput(Mb/s)", "distance_to_cell(m)"]], lsuffix="_cell2", rsuffix="_cell5")

# # Write the dataframe to a CSV file
# # df2_v_df5.to_csv(project_path / 'data' / 'output' / 'compare_cell_2_vs_cell_5_power.csv', index=False)

# # Drop the rows where the serving_cell_id_cell2 is not equal to 2 or the serving_cell_id_cell5 is not equal to 5
# df2_only = df2_v_df5[(df2_v_df5["serving_cell_id_cell2"] == 2)]
# df5_only = df2_v_df5[(df2_v_df5["serving_cell_id_cell5"] == 5)]






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

def filter_power_data(df, serving_cell_id, param):
    """
    Function that accepts a dataframe, serving cell ID, and column label as arguments.
    Filters out rows where the serving cell is equal to the provided serving_cell_id.
    Keeps the seed, sc_power(dBm), and `param` columns.
    """
    df_serving_cell = df[df["serving_cell_id"] == serving_cell_id]
    df_output_vs_param = df_serving_cell[["seed", "sc_power(dBm)", param]]
    return df_output_vs_param

# Filter the data for the interesting cell ID and parameter = "cell_throughput(Mb/s)"
df2_cell_throughput = filter_power_data(df2, 2, "cell_throughput(Mb/s)")
df5_cell_throughput = filter_power_data(df5, 5, "cell_throughput(Mb/s)")

# Every power level has 100 seeds, so we need to calculate the number of attached users for each seed and power level combination and store it in a new column
df2_cell_throughput["n_attached"] = df2_cell_throughput.groupby(["seed", "sc_power(dBm)"])['cell_throughput(Mb/s)'].transform('count')
df5_cell_throughput["n_attached"] = df5_cell_throughput.groupby(["seed", "sc_power(dBm)"])['cell_throughput(Mb/s)'].transform('count')

# Drop all duplicate rows, ignoring the index
df2_cell_throughput = df2_cell_throughput.drop_duplicates(ignore_index=True)
df5_cell_throughput = df5_cell_throughput.drop_duplicates(ignore_index=True)

 # group by sc_power(dBm) and calculate average cell throughput
result2 = df2_cell_throughput.groupby(['sc_power(dBm)', 'seed']).agg('sum')
result5 = df5_cell_throughput.groupby(['sc_power(dBm)', 'seed']).agg('sum')


# mean
result2a = result2.groupby(['sc_power(dBm)']).agg('mean')['cell_throughput(Mb/s)']
result5a = result5.groupby(['sc_power(dBm)']).agg('mean')['cell_throughput(Mb/s)']


# std
result2b = result2.groupby(['sc_power(dBm)']).agg('std')['cell_throughput(Mb/s)']
result5b = result5.groupby(['sc_power(dBm)']).agg('std')['cell_throughput(Mb/s)']


# Get the unique combinations of sc_power and cell_throughput
df2_cell_throughput = df2_cell_throughput.drop_duplicates(subset=["sc_power(dBm)", "cell_throughput(Mb/s)"])
df5_cell_throughput = df5_cell_throughput.drop_duplicates(subset=["sc_power(dBm)", "cell_throughput(Mb/s)"])



# # Get agg mean of cell throughput
# df2_cell_throughput_mean = df2_cell_throughput.groupby(["sc_power(dBm)"])["cell_throughput(Mb/s)"].agg("sum")
# df5_cell_throughput_mean = df5_cell_throughput.groupby(["sc_power(dBm)"])["cell_throughput(Mb/s)"].agg("sum")

df2_cell_throughput = df2_cell_throughput.groupby(["sc_power(dBm)"]).agg("sum")
df5_cell_throughput = df5_cell_throughput.groupby(["sc_power(dBm)"]).agg("sum")













# Join the two dataframes
df2_vs_df5_cell_throughput_mean = df2_cell_throughput.join(df5_cell_throughput, lsuffix="_cell2", rsuffix="_cell5")

# Filter the data for the interesting cell ID and parameter
df2_ue_throughput = filter_power_data(df2, 2, "ue_throughput(Mb/s)")
df5_ue_throughput = filter_power_data(df5, 5, "ue_throughput(Mb/s)")

# get the aggregate UE throughput for each seed combination
df2_ue_throughput = df2_ue_throughput.groupby(["sc_power(dBm)"]).agg("mean")["cell_throughput(Mb/s)"]
df5_ue_throughput = df5_ue_throughput.groupby(["sc_power(dBm)"]).agg("mean")["ue_throughput(Mb/s)"]



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







