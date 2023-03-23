from dataclasses import dataclass
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from kiss import Cellv2, CellEnergyModel, Sim


@dataclass(frozen=False)
class MacroCellParamsStandalone:
    """ Object for setting macro cell base station parameters."""
    p_max_dbm: float = 49.0
    power_static_watts: float = 130.0
    eta_pa: float = 0.311
    gamma_pa: float = 0.15
    power_rf_watts: float = 12.9
    power_baseband_watts: float = 29.6
    loss_feed_db: float = 3.0
    loss_dc: float = 0.075
    loss_cool: float = 0.10
    loss_mains: float = 0.09
    delta_p: float = 4.2  # 10.1109/LCOMM.2013.091213.131042
    sectors: int = 3
    antennas: int = 2

def calculate_cell_power_consumption(cell_power_dBm = 46.0):
    """
    Calculates the power consumption (in watts) of a 5G base station (gNB) based on the input cell power in dBm.
    """

    # Define the cell and interval parameters
    cell = Cellv2(sim=sim)
    interval = 1.0

    # Initialize the energy model
    energy_model = CellEnergyModel(cell, interval=interval, params=MacroCellParamsStandalone)

    # Set the cell power in dBm
    cell.set_power_dBm(cell_power_dBm)

    # Update the cell power consumption
    energy_model.update_cell_power_kW()

    # Get the cell power consumption in watts
    cell_power_consumption= energy_model.cell_power_kW

    return cell_power_consumption * 1000.0


def change_cell_power_dBm(max_cell_power_dBm):
    """
    Calculates the cell power consumption for a range of cell output powers.
    """
    # Convert input dBm to watts
    cell_power_milliwatts = 10.0 ** (max_cell_power_dBm / 10.0)
    cell_power_watts = cell_power_milliwatts / 1000.0
    print(f'Cell power: {cell_power_milliwatts:.2f} mW = {cell_power_watts:.2f} W')

    # Get a range of floats from 0 to 40 watts in 1 watt increments
    power_range_watts = np.arange(0.0, cell_power_watts, 1.0)

    # Convert the power range from watts to milliwatts
    power_range_milliwatts = power_range_watts * 1000.0

    # Convert the power range from milliwatts to dBm
    power_range_dBm = 10.0 * np.log10(power_range_milliwatts)

    # Make a list of the cell power consumption values
    AIMM_standalone = []

    # For each value in the power range, calculate the cell power consumption
    for power in power_range_dBm:
        energy_cons = calculate_cell_power_consumption(power)
        print(f'Cell output power: {power} dBm, Cell power consumption: {energy_cons} W')
        AIMM_standalone.append((energy_cons))
    
    return AIMM_standalone, power_range_dBm, power_range_watts


def change_dataclass_param(param, max_value):
    """Change the value of a dataclass parameter."""

    # Construct a a range of values from 0 to the max value in increments of 1% of the max value
    param_range = np.arange(0.0, max_value, max_value * 0.01)

    # Create a list to store the results
    AIMM_param = []

    # For each value in the range, calculate the cell power consumption
    for value in param_range:
        # Set the dataclass parameter to the current value
        MacroCellParamsStandalone.__setattr__(param, value)

        # Calculate the cell power consumption
        energy_cons = calculate_cell_power_consumption()

        # Add the result to the list
        AIMM_param.append((energy_cons))

    return AIMM_param, param_range



def set_project_path(project_path_str: str = "~/dev-02/EnergyModels/KISS"):
    # Set the project path
    project_path = Path(project_path_str).expanduser().resolve()
    project_path_str = str(project_path)
    print(f'Project path:{project_path}')
    return project_path

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

def fig_timestamp(fig, author='', fontsize=6, color='gray', alpha=0.7, rotation=0, prespace='  '):
    """
    Add a timestamp to a matplotlib figure.

    Parameters
    ----------
    fig : matplotlib Figure
        The figure to add the timestamp to.
    author : str, optional
        The author's name to include in the timestamp. Default is ''.
    fontsize : int, optional
        The font size of the timestamp. Default is 6.
    color : str, optional
        The color of the timestamp. Default is 'gray'.
    alpha : float, optional
        The transparency of the timestamp. Default is 0.7.
    rotation : float, optional
        The rotation angle of the timestamp (in degrees). Default is 0.
    prespace : str, optional
        The whitespace to prepend to the timestamp string. Default is '  '.

    Returns
    -------
    None
    """
    date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.text(
        0.01, 0.005, f"{prespace}{author} {date}",
        ha='left', va='bottom', fontsize=fontsize, color=color,
        rotation=rotation,
        transform=fig.transFigure, alpha=alpha)

def plot_and_save_results(data, labels, title, xlabel, ylabel, data_path, filename):
    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    fig_timestamp(fig, author='Kishan Sthankiya')
    for i, d in enumerate(data):
        ax.plot(d[0], d[1], label=labels[i], marker='+')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()

    # Save the figures to disk in the project folder with today's date and timestamp
    figure_path = data_path.parent / "figures"
    figure_path.mkdir(parents=True, exist_ok=True)
    today = datetime.datetime.today().strftime("%Y_%m_%d")
    now = datetime.datetime.now().strftime("%H_%M_%S")
    fig.savefig(f"{figure_path}/{today}_{now}_{filename}.png", dpi=300)



# Create a simulation
sim = Sim()

# Call the change_cell_power_dBm function
AIMM_standalone, power_range_dBm, power_range_watts = change_cell_power_dBm(43.0)


# AIMM_simulation Results
# =======================

# Set the project path
project_path = set_project_path()

# Set the data path
data_path, file_name = set_data_path("data/output/reduce_centre_cell_power/2023_03_17/rccp_s100_p43dBm", project_path)
if file_name is not None:
    data_path = data_path / file_name

# Load a csv to a dataframe
df = pd.read_csv(data_path)

# Filter the data
AIMM_sim_model, AIMM_sim_model_watts = filter_power_consumption_data(df=df, serving_cell_id=9)


# Plot the results for cell power consumption (W) vs. cell output power (dBm)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
fig_timestamp(fig, author='Kishan Sthankiya')
ax.plot(power_range_dBm, AIMM_standalone, label='AIMM standalone', marker='+')
# Set the line style to dashed
ax.plot(*zip(*AIMM_sim_model), label='AIMM simulations', linestyle='--', marker='.')
plt.xlabel('Cell output power (dBm)')
plt.ylabel('Cell power consumption (W)')
plt.title('Cell power consumption vs. cell output power')
plt.grid()
plt.legend()
plt.show()

# Plot the result for cell power consumption (W) vs. cell output power (W)
import matplotlib.pyplot as plt
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig_timestamp(fig2, author='Kishan Sthankiya')
ax2.plot(power_range_watts, AIMM_standalone, label='AIMM standalone', marker='+')
# Set the line style to dashed
ax2.plot(*zip(*AIMM_sim_model_watts), label='AIMM simulations', linestyle='--', marker='.')
plt.xlabel('Cell output power (W)')
plt.ylabel('Cell power consumption (W)')
plt.title('Cell power consumption vs. cell output power')
plt.grid()
plt.legend()
plt.show()

# Save the figures to disk in the project folder with today's date and timestamp
figure_path = data_path.parent / "figures"
figure_path.mkdir(parents=True, exist_ok=True)
today = datetime.datetime.today().strftime("%Y_%m_%d")
now = datetime.datetime.now().strftime("%H_%M_%S")
fig.savefig(f"{figure_path}/{today}_{now}_AIMM_simulation_vs_standalone_dBm.png", dpi=300)
fig2.savefig(f"{figure_path}/{today}_{now}_AIMM_simulation_vs_standalone_watts.png", dpi=300)

