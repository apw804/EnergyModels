import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from kiss import Cellv2, CellEnergyModel, Sim

def calculate_cell_power_consumption(cell_power_dBm):
    """
    Calculates the power consumption (in watts) of a 5G base station (gNB) based on the input cell power in dBm.
    """

    # Define the cell and interval parameters
    cell = Cellv2(sim=sim)
    interval = 1.0

    # Initialize the energy model
    energy_model = CellEnergyModel(cell, interval=interval)

    # Set the cell power in dBm
    cell.set_power_dBm(cell_power_dBm)

    # Update the cell power consumption
    energy_model.update_cell_power_kW()

    # Get the cell power consumption in watts
    cell_power_consumption= energy_model.cell_power_kW

    return cell_power_consumption * 1000.0



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

# Create a simulation
sim = Sim()

# Convert 46 dBm to watts
cell_power_milliwatts = 10.0 ** (46.0 / 10.0)
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



# AIMM_simulation Results
# =======================
#
# Set the project path
project_path = Path("~/dev-02/EnergyModels/KISS").expanduser().resolve()
project_path_str = str(project_path)
print(f'Project path:{project_path}')

# Set the data path
data_path = project_path / "data" / "output" / "reduce_centre_cell_power" / "2023_03_17" / "rccp_s100_p43dBm.csv"

# Load a csv to a dataframe
df = pd.read_csv(data_path)

# Filter out rows where the serving cell is the middle cell
df_centre_cell = df[df["serving_cell_id"] == 9]

# Get the top 10 rows
df_head = df_centre_cell.head(10)

# Check the number of unique values in the power consumption column
df_centre_cell["cell_power(kW)"].nunique()

# Get the rows where the power consumption is unique
df_centre_cell[df_centre_cell["cell_power(kW)"].duplicated(keep=False)]

# Keep the seed, sc_power(dBm) and cell_power(kW) columns
df_cc_output_vs_cons = df_centre_cell[["seed", "sc_power(dBm)", 
                                        "cell_power(kW)"]]

# Return rows with unique combinations of sc_power and cell_power
df_cc_output_vs_cons = df_cc_output_vs_cons.drop_duplicates(subset=[
    "sc_power(dBm)", "cell_power(kW)"
    ])

# Convert the cell_power(kW) column to watts
df_cc_output_vs_cons["cell_power(W)"] = df_cc_output_vs_cons["cell_power(kW)"] * 1e3

# Rename the columns
df_cc_output_vs_cons = df_cc_output_vs_cons.rename(columns={"sc_power(dBm)": "P_out(dBm)", "cell_power(W)": "P_cons(W)"})

# Sort the dataframe by the sc_power column
df_cc_output_vs_cons = df_cc_output_vs_cons.sort_values(by="P_out(dBm)")

# Store the P_out(dBm) and P_cons(W) columns as a list of tuples
AIMM_sim_model = list(zip(df_cc_output_vs_cons["P_out(dBm)"], df_cc_output_vs_cons["P_cons(W)"]))

# Convert the P_out(dBm) column to watts
df_cc_output_vs_cons["P_out(W)"] = 10 ** (df_cc_output_vs_cons["P_out(dBm)"] / 10.0) / 1000.0

# Store the P_out(W) and P_cons(W) columns as a list of tuples
AIMM_sim_model_watts = list(zip(df_cc_output_vs_cons["P_out(W)"], df_cc_output_vs_cons["P_cons(W)"]))




# Plot the results
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


# AIMM_simulation Results as Watts
# ================================



# Plot the results
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

