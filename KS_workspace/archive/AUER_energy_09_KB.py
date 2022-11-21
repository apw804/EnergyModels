# Kishan Sthankiya
# Basic macro cell, 5 UEs, custom UE height, logging & MME
# Scenario class is subclassed. We'll use this to randomly move UEs from their original position
# Added logging with xyz position of UE and throughput
# Increase the number of cells to 3. Attach UEs to nearest cell
# Changed run time to 20 seconds
# Average throughput added to logger
# Add QMUL energy model as function in scenario
# Add energy consumption & energy efficiency to logger
# Subclassed Cell class to include the energy model
# Add logger with pandas.DataFrame output
# 2022-10-09 20:27:31 Cell transmit behavior fix
# 2022-10-10 20:37:33 Added output plots for cell[0]

# KB Be consistent about blank lines in the file - my rule is never more than one, and never inside functions.

import os
# import modules # KB comment in wrong place (and a useless comment anyway)
from math import log10  # KB

# KB why the blank line?
import matplotlib.pyplot as plt
import numpy as np  # KB
import pandas as pd
from numpy.random import standard_normal

from AIMM_simulator_core_16 import (MME, Cell, Logger, Scenario, Sim,
                                    np_array_to_str)
from NR_5G_standard_functions_00 import Radio_state


# Subclass the Cell class to insert the qm_bs_energy_model # KB make this a docstring
class QmCell(Cell):  # KB always indent 2 spaces, not 4 (VS code setting)
    def __init__(*args, **kwargs):  # KB not sure this is needed
        Cell.__init__(s, sim, *args, **kwargs)

    def reference_signal_power(self, cell):
        """
        Calculate Reference Signal Power (dBm)
        """
        # KB never put blank lines inside functions - only between them
        # Get the maximum number of resources blocks (RBs) based on current radio state
        max_RBs = Radio_state.nPRB
        rsp = cell.power_dBm - 10 * numpy.log10(
            max_RBs * 12
        )  # KB use math.loag01 for scalars
        return rsp  # KB just return the previous expression

    def tx_dBm_per_UE(self, cell):
        """
        Calculates the transmit dBm per UE.
        Evenly divides available transmit power over the attached UEs
        """

        available_tx_dBm = cell.power_dBm - QmCell.reference_signal_power(self, cell)
        attached_UEs = cell.get_nattached()
        x = available_tx_dBm / attached_UEs
        return x  # KB just return the previous expression

    # Function for energy model
    def qm_bs_energy_model(self, cell):
        """
        QMUL energy model for each cell (base station).
        """
        carriers = cell.n_subbands

        if cell.pattern is None:
            # Then assume that the base station is 3 antennas of 120 degree (360 overall) # KB make comments fit line length
            antennas = 3
        else:
            if (
                type(cell.pattern) == np.ndarray
            ):  # KB should be "if isinstance(cell.pattern,np.array):"
                # If there's an array assume that it's 1 antenna
                antennas = 1
            else:
                antennas = 1  # assuming the function is unidirectional (for now)

        sectors = antennas
        loss_feeder = (
            np.power(10, -3 / 10) / 1000
        )  # KB what's going on here?  Is it a conversion to dB?  If so, use the provided to_dB function.  Never write cryptic expressions like this without a comment!
        tx_max_BS = (
            cell.power_dBm
        )  # KB call this tx_max_BS_dBm so we know th eunits.  ALL variables which have units should have the unit as part of the name!

        # Set values if it's a macro cell
        if tx_max_BS >= 30:
            eta_PA = 0.311
            power_RF = 12.9
            power_baseband = 29.6
            loss_DC = 0.075
            loss_cool = 0.1
            loss_mains = 0.09

        # Set values for small cell
        else:
            eta_PA = 0.067
            loss_feeder = 0
            power_RF = 1
            power_baseband = 3
            loss_DC = 0.09
            loss_cool = 0
            loss_mains = 0.11

        # Calculate the variable base station power consumption, assuming maximum load
        # KB when coding cryptic formulas like this, at least give a reference to where they've come from!  Better, add a detailed comment.
        # KB I don't like spaces around +, * etc.   It makes the formulas look too long.
        n_trx = carriers * antennas * sectors
        power_PA = lambda tx_max_BS: tx_max_BS / (eta_PA * (1 - loss_feeder))
        power_BS_sum = power_PA(tx_max_BS=tx_max_BS) + power_RF + power_baseband
        power_BS_loss = (1 - loss_DC) * (1 - loss_mains) * (1 - loss_cool)
        power_BS_frac = power_BS_sum / power_BS_loss
        power_BS_var = n_trx * power_BS_frac

        power_BS_static = n_trx * (
            power_PA(0) + power_RF + power_baseband / power_BS_loss
        )

        power_BS = power_BS_static + power_BS_var

        return power_BS  # KB just return the previous expression

    def energy(self, cell):
        self.energy_total += self.interval * (self.qm_bs_energy_model(cell))
        return self.energy_total


# Subclass Scenario and use the 'loop' method as specified in the AIMM documentation
class QmScenario(Scenario):
    # This loop sets the amount of time between each event
    def loop(s, interval=0.1):
        while True:
            for ue in s.sim.UEs:
                # Here we update the position of the UEs in each iteration of the loop ()
                ue.xyz[:2] += 20 * standard_normal(2)
                yield s.sim.wait(interval)


"""  for cell in s.sim.cells:
    if len(s.sim.cells[cell.i].attached) == 0:
        min = QmCell.reference_signal_power(s, cell)
        cell.set_power_dBm(min)
    else:
        scaled_tx_power = QmCell.reference_signal_power(
            s, cell
        ) + QmCell.tx_dBm_per_UE(s, cell)
        cell.set_power_dBm(scaled_tx_power) """


# Write a new logger from the perspective of the the cells. # KB "the" is repeated
# I want the time, cell ID, number of UEs attached, throughput of all attached UEs, EC & EE
class QmEnergyLogger(Logger):

    # Constructor for QmEnergyLogger with additional parameter
    def __init__(s, *args, **kwargs):
        # Calling the parent constructor using ClassName.__init__()
        Logger.__init__(s, *args, **kwargs)
        # Adding the cols and main_dataframe
        s.cols = (
            "time",
            "cell",
            "n_UEs",
            "a_tp(Mbps)",
            "EC(kW)",
            "EE",
        )  # KB I prefer single quotes around strings - it looks cleaner
        s.main_dataframe = pd.DataFrame(data=None, columns=list(s.cols))

    def append_row(s, new_row):
        temp_df = pd.DataFrame(data=[new_row], columns=list(s.cols))
        s.main_dataframe = pd.concat([s.main_dataframe, temp_df])

    def loop(s):
        # Write to stdout
        s.f.write("#time\tcell\tn_UEs\ta_tp (Mb/s)\tEC (kW)\t\tEE (Mbps/kW)\n")
        while True:
            # Needs to be per cell in the simulator
            for cell in s.sim.cells:
                # Get timestamp
                tm = s.sim.env.now
                # Get cell ID
                cell_i = cell.i
                # Get number of UEs attached to this cell
                n_UEs = len(s.sim.cells[cell_i].attached)
                # Get total throughput of attached UEs
                tp = 0.00
                for ue_i in s.sim.cells[cell_i].attached:
                    tp += s.sim.cells[cell_i].get_UE_throughput(ue_i)
                ec = (
                    QmCell.qm_bs_energy_model(s, cell) / 1000
                )  # KB what is 1000 there for?   NEVER put "magic numbers" like this without explanation!
                # Calculate the energy efficiency
                if tp == 0:
                    ee = 0  # KB think about types - should this be 0.0?
                else:
                    ee = tp / ec

                # Write to stdout
                s.f.write(
                    f"t={tm:.1f}\t{cell_i}\t{n_UEs}\t{tp:.2f}\t\t{ec:.2f}\t\t{ee:.3f}\n"
                )  # KB why two tabs?

                # Write these variables to the main_dataframe
                row = (tm, cell_i, n_UEs, tp, ec, ee)
                new_row = [np.round(each_arr, decimals=3) for each_arr in row]
                s.append_row(new_row)

            yield s.sim.wait(s.logging_interval)

    def finalize(s):
        cwd = os.getcwd()
        path = cwd + os.path.basename(__file__)
        s.main_dataframe.to_csv(path)

        # Seperate out into dataframes for each cell[i] # KB spelling error - should be "Separate"
        df = s.main_dataframe
        df_cell0 = df[df.cell.eq(0)]
        df_cell1 = df[df.cell.eq(1)]
        df_cell2 = df[df.cell.eq(2)]

        # Cell0 plot # KB better to always keep plotting separate from everything else
        # x-axis is time
        x = df_cell0["time"]
        y_0_tp = df_cell0["a_tp(Mbps)"]
        y_0_ec = df_cell0["EC(kW)"]
        y_0_ee = df_cell0["EE"]
        y_0_UEs = df_cell0["n_UEs"]

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(4)

        ax0.plot(x, y_0_tp, "tab:green")
        ax1.plot(x, y_0_ec, "tab:green")
        ax2.plot(x, y_0_ee, "tab:blue")
        ax3.plot(x, y_0_UEs, "tab:red")

        ax0.set(ylabel="Throughput (Mbps)")
        ax1.set(ylabel="Power (kW)")
        ax2.set(ylabel="Energy Efficiency (Mbps/kW)")
        ax3.set(xlabel="time (s)", ylabel="Number of attached UEs")
        plt.xlim([0, max(x) + 0.5])
        plt.show()
        # print(s.main_dataframe)


def QMUL_09():
    # Create simulator instance
    sim = Sim()
    # Create cells (accept default= macro cell, set verbosity to 1)
    for i in range(3):
        cell = sim.make_cell(verbosity=1)

    # Create 5 UEs. Default attachment is simple pathloss.
    for i in range(5):
        # Create the UE
        sim.make_UE().attach_to_nearest_cell()
    # Add logger
    sim.add_logger(QmEnergyLogger(sim, logging_interval=1))
    # Apply custom scenario
    sim.add_scenario(QmScenario(sim))
    # Add Mobility Management Engine
    sim.add_MME(MME(sim, interval=10.0))
    # Run for 100 seconds
    sim.run(until=100)


# KB ALWAYS use "if __name__=='__main__':" here
QMUL_09()
