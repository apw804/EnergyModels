# A 'simple' script to reproduce the Holtkamp et al. (2013) results.
# DOI: 10.1109/LCOMM.2013.091213.131042

# P_supply = supply power consumption of a BS
# P0 = load-independent power consumption
# P1 = maximum supply power consumption
# delta_p = linear gradient of power consumption
# P_max = maximum total transmission power (watts)
# M_sec = number of sectors
# P_sleep = power consumption in sleep mode

# W = transmission bandwidth (Hz)
# D = number of BS radio chains/antennas

# P_PA = power amplifier power consumption
# P_RF = RF power consumption
# P_BB = baseband processing power consumption

# sigma_feed = feedline loss (dB)
# sigma_DC = DC power loss (dB)
# sigma_MS = Mains Supply power loss (dB)
# sigma_cool = cooling power loss (dB)

from dataclasses import dataclass
import datetime
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Macro:
    """Macro cell parameters."""
    P_PA_limit:     float =  80.0    # watts
    eta_PA_max:     float =   0.36
    gamma:          float =   0.15
    P_prime_BB:     float =  29.4    # watts
    P_prime_RF:     float =  12.9    # watts
    sigma_feed:     float =   0.50
    sigma_DC:       float =   0.075
    sigma_cool:     float =   0.100
    sigma_MS:       float =   0.075
    M_sec:            int =   3
    P_max:          float =  40.0    # watts
    P1:             float = 460.4    # watts
    delta_p_10MHz:  float =   4.2
    P_sleep_zero:   float = 324.0    # watts


def P_sleep(D, P_sleep_zero):
    """
    Returns the power consumption of the base station in sleep mode.

    Parameters
    ----------
    model : str
        The model to use for the calculation.
    D : int
        Number of BS radio chains/antennas
    P_sleep_zero : float
        Power consumption of the base station in sleep mode (watts). 
        This is a "reference value for the single antenna BS chosen such that 
        P_sleep matches the complex model value for two antennas" 
        (Holtkamp et al. 2013).
    """
    return D * P_sleep_zero


def eta_PA(eta_PA_max, gamma, P_PA_limit, P_max, D):
    """
    Returns the power amplifier efficiency.
    """

    eta_PA = eta_PA_max * (1 - gamma * np.log2 (P_PA_limit / (P_max / D)))
    return eta_PA


def P_PA(P_max: float, D: int, eta_PA: float, sigma_feed: float):
    """
    Returns the power amplifier power consumption of the base station in watts.
    """
    return P_max / D * eta_PA * (1 - sigma_feed)


def P_RF(D: int = 1, W: int = 10e3, P_prime_RF: float = Macro.P_prime_RF):
    """
    Returns the RF power consumption of the base station in watts.

    Parameters
    ----------
    D : int
        Number of BS radio chains/antennas
    W : int
        Transmission bandwidth (Hz)
    P_prime_RF : float
        RF power consumption (watts)
      
    Returns
    -------
    float
        RF power consumption of the base station in watts.
    """

    return  D * (W / 10e3) * P_prime_RF


def P_BB(D: int = 1, W: int = 10e3, P_prime_BB: float = Macro.P_prime_BB):
    """
    Returns the baseband processing power consumption of the base station 
    in watts.
    """
    return  D * (W / 10e3) * P_prime_BB


def P1(D: int, W: int = 10e3):
    """
    Returns the complex model maximum supply power consumption of the 
    base station  per antenna / radio chain in watts.

    Parameters
    ----------
    D : int
        Number of BS radio chains/antennas
    W : int
        Transmission bandwidth (Hz)
    
    Returns
    -------
    float
        Complex model maximum supply power consumption of the base station, per 
         antenna / radio chain in watts.
    """
    BB = P_BB(D, W, P_prime_BB = Macro.P_prime_BB)
    RF = P_RF(D, W, P_prime_RF = Macro.P_prime_RF)
    PA = P_PA(P_max = Macro.P_max,
              D=D,
              eta_PA = eta_PA(eta_PA_max = Macro.eta_PA_max,
                              gamma = Macro.gamma,
                              P_PA_limit = Macro.P_PA_limit,
                              P_max = Macro.P_max,
                              D=D),
              sigma_feed = Macro.sigma_feed)
    
    o_DC   = Macro.sigma_DC
    o_MS   = Macro.sigma_MS
    o_cool = Macro.sigma_cool

    numerator = BB + RF + PA
    denominator = (1 - o_DC) * (1 - o_MS) * (1 - o_cool) 

    return numerator / denominator


def param_P_model(M_sec, P_1, delta_p, P_max, chi=1.0):
    """
    Returns the parameterised model maximum supply power consumption of the base
    station in watts.

    Parameters
    ----------
    M_sec : int
        Number of sectors
    P_1 : float
        Maximum supply power consumption (watts)
    delta_p : float
        Linear gradient of power consumption
    P_max : float
        Maximum total transmission power (watts)
    chi : float
        Load factor (default = 1.0)

    Returns
    -------
    float
        Parameterised model maximum supply power consumption of the base station
        in watts.
    """
    return M_sec * (P_1 + delta_p * P_max * (chi - 1.0))


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


if __name__ == "__main__":



    # Create a list of loads
    loads = np.arange(0, 1.01, 0.01).astype(float)

    # Run param_P_model for each load
    # Create a list to hold the results
    param_model = []
    for load in loads:
        p_cons = param_P_model(M_sec = Macro.M_sec, 
                               P_1= Macro.P1, 
                               delta_p = Macro.delta_p_10MHz, 
                               P_max=Macro.P_max, 
                               chi=load)
        
        # P_out is the product of the load and the maximum transmission power, in watts
        P_out_watts = load * Macro.P_max

        # Convert P_out_watts to dBm
        P_out_dBm = 10 * np.log10(P_out_watts / 1e-3)

        # Print the results
        print(f"[SUPPLY] Load: {P_out_dBm:.2f}, Power cons.:{p_cons:.4f} W")

        # Append the load and result to the list
        param_model.append((P_out_dBm, p_cons))


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

    # Store the results in a list
    AIMM_model = []
    for index, row in df_cc_output_vs_cons.iterrows():
        AIMM_model.append((row["P_out(dBm)"], row["P_cons(W)"]))

    # Get both the AIMM and Holtkamp models as dataframes
    df_AIMM = pd.DataFrame(AIMM_model, columns=["P_out(dBm)", "P_cons(W)"])
    df_Holtkamp = pd.DataFrame(param_model, columns=["P_out(dBm)", "P_cons(W)"])

    # Plot the dBm vs W results
    fig1, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_AIMM["P_out(dBm)"], df_AIMM["P_cons(W)"], label="AIMM")
    ax.plot(df_Holtkamp["P_out(dBm)"], df_Holtkamp["P_cons(W)"], label="Holtkamp")
    ax.set_xlabel("Output power (dBm)")
    ax.set_ylabel("Power consumption (W)")
    ax.set_title("AIMM vs Holtkamp model")
    ax.grid()
    # Set markers on the plot
    ax.plot(df_AIMM["P_out(dBm)"], df_AIMM["P_cons(W)"], ".", label="AIMM")
    ax.plot(df_Holtkamp["P_out(dBm)"], df_Holtkamp["P_cons(W)"], ".", label="Holtkamp")
    # Add the figure_timestamp to the plot
    fig_timestamp(fig=fig1, author='Kishan Sthankiya')
    ax.legend()


    # Convert the P_out(dBm) column to watts and store in a new column
    df_AIMM["P_out(W)"] = 10 ** (df_AIMM["P_out(dBm)"] / 10) * 1e-3
    df_Holtkamp["P_out(W)"] = 10 ** (df_Holtkamp["P_out(dBm)"] / 10) * 1e-3

    # Plot the W vs W results
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_AIMM["P_out(W)"], df_AIMM["P_cons(W)"], label="AIMM")
    ax.plot(df_Holtkamp["P_out(W)"], df_Holtkamp["P_cons(W)"], label="Holtkamp")
    ax.set_xlabel("Output power (W)")
    ax.set_ylabel("Power consumption (W)")
    ax.set_title("AIMM vs Holtkamp model")
    ax.grid()
    # Set markers on the plot
    ax.plot(df_AIMM["P_out(W)"], df_AIMM["P_cons(W)"], ".", label="AIMM")
    ax.plot(df_Holtkamp["P_out(W)"], df_Holtkamp["P_cons(W)"], ".", label="Holtkamp")
    # Add the figure_timestamp to the plot
    fig_timestamp(fig=fig1, author='Kishan Sthankiya')
    ax.legend()


    # Make the x axis start from 0
    ax.set_xlim(left=0)
    plt.show()

    # Save the figures to disk in the project folder with today's date and timestamp
    figure_path = data_path.parent / "figures"
    figure_path.mkdir(parents=True, exist_ok=True)
    today = datetime.datetime.today().strftime("%Y_%m_%d")
    now = datetime.datetime.now().strftime("%H_%M_%S")
    fig1.savefig(f"{figure_path}/{today}_{now}_AIMM_vs_Holtkamp_dBm_vs_W.png", dpi=300)
    fig2.savefig(f"{figure_path}/{today}_{now}_AIMM_vs_Holtkamp_W_vs_W.png", dpi=300)



