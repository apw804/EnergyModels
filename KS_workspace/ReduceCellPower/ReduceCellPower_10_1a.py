# Kishan Sthankiya
# Basic energy model class based on test_f_callback_00.py
# Uses callback functions to compute the energy.
# Added dataclasses for small cells and macro cells.
# Added bounding box with shapely.geometry.
# 2022-11-20 13:55 Add custom QmEnergyLogger
# Add cell power_dBm level to logger
# Add hexgrid functions!!!
# Add average spectral efficiency to logger
# Add QmScenario class.
# Cleanup imports as per KB suggestions (2022-11-22)
# Updated hexgrid function using hexalattice
# Updated PPP function
# Correction for random number generator used in PPP function, and attached to simulator rng
# Amended radius_polar equation from incorrect random.uniform to random.exponential
# Added checks for PPP points that exceed the sim_radius when converted to cartesian coords.
# Move to .tsv files for output of logger as easier to understand!
# Ensure a single cell and single UE.
# Single UE reporting interval is scaled to 1/1000 of the `until` time
# Increase the number of subbands to 12
# Subbands decreased to 1
# Function to output a file of the input params added
# Updated log file location and name
# Disabled AIMM profiler in test_01() by removing "params={'profile': 'SimLogProfile'}" from Sim object
# Added cleaner Cell Only logging function
# Outputs fpr cellLog, EnergyLog and ConfigLog now written to files.
# Adding a UE logger.
# Fixed the 'until' problem. Credit to Richard who pointed out the @SmallCellParameters capitalisation issue.
# v10 - now re-enables the number of UEs as a command line argument.
# v10.1 - 1) Removed the QmScenarioReduceCellPower class; 2) Fixed EnergyLog output to include time t=0;
# 3) fixed list comprehensions for the EnergyLogger
# FIXME - energy cons values do not vary with cell load and do not look right


import argparse
import json
import sys
from dataclasses import dataclass
from operator import itemgetter
from os import getcwd
from pathlib import Path
from sys import stdout
from time import strftime, localtime
from typing import Literal, get_args

import pandas as pd
from AIMM_simulator import Cell, UE, Scenario, Sim, from_dB, Logger, np_array_to_str, CQI_to_64QAM_efficiency, \
    NR_5G_standard_functions
from hexalattice.hexalattice import *
import numpy as np
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
INFO_LOG = True
DEBUG_LOG = False


@dataclass(frozen=True)
class SmallCellParameters:
    """ Object for setting small cell base station parameters."""
    p_max_db: float = 23.0
    power_static_watts: float = 6.8
    eta_pa: float = 0.067
    power_rf_watts: float = 1.0
    power_baseband_watts: float = 3.0
    loss_feed_db: float = 0.00
    loss_dc_db: float = 0.09
    loss_cool_db: float = 0.00
    loss_mains_db: float = 0.11
    delta_p: float = 4.0  # 10.1109/LCOMM.2013.091213.131042


@dataclass(frozen=True)
class MacroCellParameters:
    """ Object for setting macro cell base station parameters."""
    p_max_db: float = 49.0
    power_static_watts: float = 75.0
    eta_pa: float = 0.311
    power_rf_watts: float = 12.9
    power_baseband_watts: float = 29.6
    loss_feed_db: float = -3.0
    loss_dc_db: float = 0.075
    loss_cool_db: float = 0.10
    loss_mains_db: float = 0.0
    delta_p: float = 4.2  # 10.1109/LCOMM.2013.091213.131042


class CellEnergyModel:
    """
    Defines a complete self-contained system energy model for a 5G base station (gNB).

    Parameters
    ----------
    cell : Cell
        Cell instance which this model attaches to.
    interval: float
        Time interval between CellEnergyModel updates.

    """

    def __init__(s,
                 cell: Cell,
                 interval=1.0):
        """
        Initialize variables.
        """
        # Check the cell power_dBm when initialising. Set as macro or small cell accordingly.
        s.cell = cell
        logging.info("Attaching CellEnergyModel to Cell[%s]", s.cell.i)
        s.POWER_DBM_MAX = s.cell.power_dBm
        if not 0. < s.POWER_DBM_MAX < 30.0:
            logging.info("Cell[%s] transmit power > 30 dBm.", s.cell.i)

            s.cell_type = 'MACRO'
            logging.info("Cell[%s] type set to %s.", cell.i, s.cell_type)

            s.params = MacroCellParameters()
            logging.info("Cell[%s] params set to %s.", cell.i, s.params)

        else:
            logging.info("Cell[%s] transmit power < 30 dBm.", s.cell.i)

            s.cell_type = 'SMALL'
            logging.info("Cell[%s] type set to %s.", s.cell.i, s.cell_type)

            s.params = SmallCellParameters()
            logging.info("Cell[%s] params set to %s.", s.cell.i, s.params)

        if not isinstance(s.cell.pattern, list):
            s.cell_antennas = 3  # Assume 3*120 degree antennas
        else:
            s.cell_antennas = 1  # If an array or function, assume unidirectional (for now)

        s.cell_power_static_kW = s.params.power_static_watts * 1.e-3  # baseline energy use in kilowatts

        if cell.sim.env.now == 0.:
            s.cell_power_kW = s.cell_antennas * s.cell_power_static_kW  # Instantaneous independent power in kilowatts
        else:
            s.cell_power_kW = s.cell_power_static_kW + s.update_cell_power()
        s.cell_power_kW_Hx = np.array(s.cell_power_kW)  # FIXME - History of cell power levels

    def update_cell_power(s):
        """
        Returns the power consumption (in watts), per transceiver.
        """

        # Frame = 10 milliseconds. Sub-frame = 1 milliseconds
        # Bandwidth of 1 sub-carrier = 30kHz
        # 1 Resource block (PRB) = 12 sub-carriers = 12 * 30 kHz = 360 kHz
        nRB_sc = NR_5G_standard_functions.Radio_state.NRB_sc
        sc_bw_kHz = 30.0  # sub-carrier bandwidth
        # Convert the cell bandwidth to kHz and find the max RBs this channel can support respecting the SCS for the
        # radio state numerology
        number_RBs_possible = s.cell.bw_MHz * 1000 // ((nRB_sc * sc_bw_kHz) + sc_bw_kHz)
        channel_tx_bandwidth_kHz = number_RBs_possible * nRB_sc * sc_bw_kHz
        channel_tx_bandwidth_MHz = channel_tx_bandwidth_kHz / 1000

        # Get the constant value for baseband power from s.params.power_baseband_watts.
        # Calculate baseband power cons. in watts.
        power_baseband_watts = s.cell_antennas * (
                channel_tx_bandwidth_MHz / s.cell.bw_MHz) * s.params.power_baseband_watts

        # Get the constant value for RF power from s.params.power_rf_watts.
        # Calculate RF power cons. in watts.
        power_rf_watts = s.cell_antennas * (channel_tx_bandwidth_MHz / s.cell.bw_MHz) * s.params.power_rf_watts

        if s.cell.power_dBm <= s.params.p_max_db:
            # Get the PA efficiency for the given power level
            gamma_pa = 0.15  # Ref: 10.1109/LCOMM.2013.091213.131042
            eta_pa_scaled = s.params.eta_pa * (
                    1 - gamma_pa * np.log2(from_dB(s.POWER_DBM_MAX) / (from_dB(s.cell.power_dBm) / s.cell_antennas)))

            # The power amplifier (PA) power consumption scales with the maximum transmission power per antenna,
            # (P_max / D), the PA efficiency (eta_pa) and feeder cable losses.
            power_pa_watts = from_dB(s.cell.power_dBm) / s.cell_antennas * eta_pa_scaled * (
                    1 - from_dB(s.params.loss_feed_db))

            # Get the value of the losses (denominator)
            power_losses = (1 - s.params.loss_dc_db) + (1 - s.params.loss_mains_db) + (1 - s.params.loss_cool_db)

            # Get the power consumption (numerator) total
            power_consumption = power_baseband_watts + power_rf_watts + power_pa_watts

            # Get the power output
            power_out = power_consumption / power_losses

            # Calculate the linear approximation of the cell power model
            s.cell_power_kW = s.cell_antennas * (s.params.power_static_watts + s.params.delta_p * power_out)

        if s.cell.power_dBm > s.params.p_max_db:
            raise ValueError('Power cannot exceed the maximum transceiver power!')

    def get_cell_power_kW(s):
        return s.cell_power_kW

    def f_callback(s, x, **kwargs):
        # print(kwargs)
        if isinstance(x, Cell):
            # print(f't={s.sim.env.now:.1f}: cell[{x.i}] (check from kwargs: {kwargs["cell_i"]})
            s.update_cell_power()


# End class Energy


class SimLogger(Logger):
    """
       Custom Logger for energy modelling.
       """

    def __init__(s, sim, seed, until, sim_args=None, func=None, header='', f=stdout,
                 logging_interval=1.0):
        s.until: float = until
        s.seed: int = seed
        super(SimLogger, s).__init__(sim, func, header, f, logging_interval, np_array_to_str=np_array_to_str)
        s.sim_args = sim_args

    _LOGTYPES = Literal["Cell", "UE", "Energy", "Config", "PerfProfile"]

    @staticmethod
    def date_str():
        return strftime('%Y-%m-%d', localtime())

    @staticmethod
    def time_str():
        return strftime('%H:%M:%S', localtime())

    @staticmethod
    def log_folder():
        script_parent_dir = Path(__file__).resolve().parents[1]
        logging_path = str(script_parent_dir) + '/logfiles/' + str(Path(__file__).stem)
        today_folder = Path(logging_path + '/' + SimLogger.date_str())
        today_folder.mkdir(parents=True, exist_ok=True)
        return str(today_folder)

    @classmethod
    def write_log(cls, logtype_: _LOGTYPES, sim_args_dict: dict = None, df: pd.DataFrame = None):
        if logtype_ is not None:
            options = get_args(cls._LOGTYPES)
            assert logtype_ in options, f"'{logtype_}' is not in {options}"
            logfile_name = str(Path(__file__).stem) + '_' + cls.time_str() + '_' + logtype_ + 'Log'
            filepath = SimLogger.log_folder() + '/' + logfile_name
            if logtype_ == "Config":
                filepath = filepath + ".json"
                with open(filepath, 'w') as f:
                    json.dump(sim_args_dict, f)
            elif logtype_ == "PerfProfile":
                pass  # FIXME - figure out how to write the CPU + RAM profiles
            else:
                df.to_csv(str(filepath) + '.tsv', index=False, sep='\t', na_rep='NaN', header=True,
                          float_format='%g')
            return print(f'Logfile written to {filepath}')
        else:
            raise Exception(f"A logfile type MUST be specified. E.g. {get_args(cls._LOGTYPES)}")

    def finalize(s):
        # Write the args used for this experiment to json for reproducing
        s.write_log(sim_args_dict=s.sim_args, logtype_="Config")


# END class SimLogger

class CellLogger(Logger):
    """
    Custom Logger for energy modelling.
    """

    def __init__(s, sim, until, sim_args=None, func=None, header='', f=stdout,
                 logging_interval=1.0):

        s.until: float = until
        s.cell_dataframe = None  # Create empty placeholder for later pd.DataFrame
        super(CellLogger, s).__init__(sim, func, header, f, logging_interval, np_array_to_str=np_array_to_str)
        s.sim_args = sim_args

    def get_cell_data(s):
        # Create a dictionary of the variables we are interested in
        data = {
            "time": [s.sim.env.now] * s.sim.get_ncells(),
            "cell_id": [k.i for k in s.sim.cells],
            "cell_xyz": [str(k.xyz).strip('[] ') for k in s.sim.cells],
            "cell_subbands": [k.n_subbands for k in s.sim.cells],
            "n_attached_ues": [k.get_nattached() for k in s.sim.cells],
            "power_dBm": [k.power_dBm for k in s.sim.cells],
            "avg_pdsch_tput_Mbps": [k.get_average_throughput() for k in s.sim.cells]
        }
        return data

    def initialise_dataframe(s):
        """
        Create CellLogger DataFrame and populate with first batch of data.
        """
        data = s.get_cell_data()
        s.cell_dataframe = pd.DataFrame.from_dict(data)

    def add_to_cell_dataframe(s):
        temp_df = pd.DataFrame.from_dict(s.get_cell_data())
        old_new_df = [s.cell_dataframe, temp_df]
        s.cell_dataframe = pd.concat(old_new_df)

    # noinspection PyPep8Naming
    def loop(s):
        while True:
            # if there isn't a dataframe, initialise it!
            if s.cell_dataframe is None:
                s.initialise_dataframe()

            # if there is already a dataframe, add the cell data to it.
            else:
                s.add_to_cell_dataframe()
            yield s.sim.wait(s.logging_interval)

    def finalize(s):
        print(s.cell_dataframe)
        SimLogger.write_log(logtype_="Cell", df=s.cell_dataframe)


# END class CellLogger


class UeLogger(Logger):
    """
    Custom Logger for UE data.
    """

    def __init__(s, sim, until, sim_args=None, func=None, header='', f=stdout,
                 logging_interval=1.0):

        s.until: float = until
        s.ue_dataframe = None  # Create empty placeholder for later pd.DataFrame
        super(UeLogger, s).__init__(sim, func, header, f, logging_interval, np_array_to_str=np_array_to_str)
        s.sim_args = sim_args

    def get_ue_data(s):
        # Create a dictionary of the variables we are interested in
        data = {
            "time": [s.sim.env.now] * s.sim.get_nues(),
            "ue_id": [k.i for k in s.sim.UEs],
            "ue_xyz": [str(k.xyz).strip('[] ') for k in s.sim.UEs],
            "ue_reporting_interval": [k.reporting_interval for k in s.sim.UEs],
            "ue_serving_cell_id": [k.serving_cell.i for k in s.sim.UEs],
            "ue_serving_cell_rsrp": [k.serving_cell.get_rsrp(k.i) for k in s.sim.UEs],
            "background_thermal_noise_dBm": [k.noise_power_dBm for k in s.sim.UEs],
            "pathloss_model": [k.pathloss.__class__.__name__ for k in s.sim.UEs],
            "pathloss_dBm": [k.pathloss(k.serving_cell.xyz, k.xyz) for k in s.sim.UEs],
            # Not sure this one is going to work
            # maybe try without the brackets or
            # find out what the __call__ method does)
            "ue_neighbouring_cell_interference_dBm": "TBC (AIMM v2.0.2)",
            # Use the SINR to reverse engineer the interference
            "ue_sinr_dB": "TBC (AIMM v2.0.2)",
            "ue_cqi": [k.serving_cell.get_UE_CQI(k.i).item() for k in s.sim.UEs],
            "ue_pdsch_tput_Mbps": [k.serving_cell.get_UE_throughput(k.i) for k in s.sim.UEs]
        }
        return data

    def initialise_dataframe(s):
        """
        Create UELogger DataFrame and populate with first batch of data.
        """
        data = s.get_ue_data()
        s.ue_dataframe = pd.DataFrame.from_dict(data)

    def add_to_ue_dataframe(s):
        temp_df = pd.DataFrame.from_dict(s.get_ue_data())
        old_new_df = [s.ue_dataframe, temp_df]
        s.ue_dataframe = pd.concat(old_new_df)

    # noinspection PyPep8Naming
    def loop(s):
        while True:
            # if there isn't a dataframe, initialise it!
            if s.ue_dataframe is None:
                s.initialise_dataframe()

            # if there is already a dataframe, add the cell data to it.
            else:
                s.add_to_ue_dataframe()
            yield s.sim.wait(s.logging_interval)

    def finalize(s):
        print(s.ue_dataframe)
        SimLogger.write_log(logtype_="UE", df=s.ue_dataframe)


# END class UeLogger


class EnergyLogger(Logger):
    """
    Custom Logger for energy modelling.
    """

    def __init__(s, sim, seed, cell_energy_models, until, func=None, header='', f=stdout,
                 logging_interval=1.0):
        s.cell_energy_models = cell_energy_models
        s.until: float = until
        s.energy_dataframe = None  # Placeholder for later pd.DataFrame
        super(EnergyLogger, s).__init__(sim, func, header, f, logging_interval, np_array_to_str=np_array_to_str)

    def get_energy_data(s):

        def get_pdsch_throughput_Mbps(cell):
            """ Gets the sum total of downlink throughput for a given cell."""

            # Fetch the cell throughput_Mbps reports
            reports = cell.reports['throughput_Mbps']

            # Get a list of UE's that are attached to this cell
            attached_ue_ids = [ue_id for ue_id in cell.attached]

            # Get a list of the 'throughput_Mbps' values for attached UEs for the current time period
            attached_ue_tp_reports = np.array([reports[ue_id][1] for ue_id in attached_ue_ids if
                                               ue_id in reports])

            # Sum the throughput
            pdsch_tp_Mbps = attached_ue_tp_reports.sum()

            return pdsch_tp_Mbps

        def get_power_kW(cell: Cell):
            """Get the energy model object for this cell"""

            # Get the CellEnergyModel object from the dictionary for this cell
            cell_em = s.cell_energy_models[cell.i]

            return cell_em.update_cell_power_kW()

        def get_cell_ee(cell: Cell):
            """ Get the cell Energy Efficiency (bits/second/watt)"""

            # Convert the throughput to bits/s
            pdsch_tp_bps = get_pdsch_throughput_Mbps(cell) * 1.e6

            # Convert power consumption to watts
            cell_power_cons_W = get_power_kW(cell) * 1.e3

            return pdsch_tp_bps / cell_power_cons_W

        def get_cell_se(cell: Cell):
            """ Get the cell Spectral Efficiency (bits/second/Hertz)"""

            return get_pdsch_throughput_Mbps(cell) / cell.bw_MHz * 1.e6

        # Create a dictionary of the variables we are interested in
        data = {
            "time": [s.sim.env.now] * s.sim.get_ncells(),
            "cell_id": [k.i for k in s.sim.cells],
            "cell_tx_power_dBm": [k.power_dBm for k in s.sim.cells],
            "cell_nattached_ues": [k.get_nattached() for k in s.sim.cells],
            "pdsch_throughput_Mbps": [get_pdsch_throughput_Mbps(k) for k in s.sim.cells],
            "cell_power_cons_kW": [get_power_kW(k) for k in s.sim.cells],
            "cell_EE_bps/W": [get_cell_ee(k) for k in s.sim.cells],
            "cell_SE_bps/Hz": [get_cell_se(k) for k in s.sim.cells]
        }
        return data

    def initialise_dataframe(s):
        """
        Create EnergyLogger DataFrame and populate with first batch of data.
        """
        data = s.get_energy_data()
        s.energy_dataframe = pd.DataFrame.from_dict(data)

    def add_to_energy_dataframe(s):
        temp_df = pd.DataFrame.from_dict(s.get_energy_data())
        old_new_df = [s.energy_dataframe, temp_df]
        s.energy_dataframe = pd.concat(old_new_df)

    # noinspection PyPep8Naming
    def loop(s):
        while True:
            # if there isn't a dataframe, initialise it!
            if s.energy_dataframe is None:
                s.initialise_dataframe()

            # if there is already a dataframe, add the cell data to it.
            else:
                s.add_to_energy_dataframe()
            yield s.sim.wait(s.logging_interval)

    def finalize(s):
        print(s.energy_dataframe)
        SimLogger.write_log(logtype_="Energy", df=s.energy_dataframe)


# END class QmEnergyLogger


def generate_ppp_points(sim, expected_pts=100, sim_radius=500.0):
    """
    Generates npts points, distributed according to a homogeneous PPP
    with intensity lamb and returns an array of distances to the origin.
    """
    sim_rng = sim.rng

    # Simulation window parameters
    sim_radius = sim_radius  # radius of disk
    xx0 = 0
    yy0 = 0  # centre of disk
    areaTotal = np.pi * sim_radius ** 2  # area of disk

    # Point process parameters
    lambda0 = expected_pts / areaTotal  # intensity (ie mean density) of the Poisson process

    # Simulate Poisson point process
    numbPoints = sim_rng.poisson(lambda0 * areaTotal)  # Poisson number of points
    theta = 2 * np.pi * sim_rng.uniform(0, 1, numbPoints)  # angular coordinates
    rho = sim_radius * np.sqrt(sim_rng.uniform(0, 1, numbPoints))  # radial coordinates

    # Convert from polar to Cartesian coordinates
    xx = rho * np.cos(theta)
    yy = rho * np.sin(theta)

    # Shift centre of disk to (xx0,yy0)
    xx = xx + xx0
    yy = yy + yy0
    return np.column_stack((xx, yy))


def hex_grid_setup(origin: tuple = (0, 0), isd: float = 500.0, sim_radius: float = 1000.0):
    """
    Creates a hex grid of 19 sites (57 sectors) using the hexalattice module when using the defaults.
    Returns the
    """
    fig, ax = plt.subplots()

    hexgrid_xy, _ = create_hex_grid(nx=5,
                                    ny=5,
                                    min_diam=isd,
                                    crop_circ=sim_radius,
                                    align_to_origin=True,
                                    edge_color=[0.75, 0.75, 0.75],
                                    h_ax=ax,
                                    do_plot=True)

    hexgrid_x = hexgrid_xy[:, 0]
    hexgrid_y = hexgrid_xy[:, 1]
    circle_dashed = plt.Circle(origin, sim_radius, fill=False, linestyle='--', color='r')

    ax.add_patch(circle_dashed)
    ax.scatter(hexgrid_x, hexgrid_y, marker='2')
    ax_scaling = 3 * isd + 500  # Factor to set the x,y-axis limits relative to the isd value.
    ax.set_xlim([-ax_scaling, ax_scaling])
    ax.set_ylim([-ax_scaling, ax_scaling])
    ax.set_aspect('equal')
    return hexgrid_xy, fig


def fig_timestamp(fig, author='', fontsize=6, color='gray', alpha=0.7, rotation=0, prespace='  '):
    # Keith Briggs 2020-01-07
    # https://riptutorial.com/matplotlib/example/16030/coordinate-systems-and-text
    date = strftime('%Y-%m-%d %H:%M', localtime())
    fig.text(  # position text relative to Figure
        0.01, 0.005, prespace + '%s %s' % (author, date,),
        ha='left', va='bottom', fontsize=fontsize, color=color,
        rotation=rotation,
        transform=fig.transFigure, alpha=alpha)


def test_01(seed=0, subbands=1, isd=5000.0, sim_radius=2500.0, nues=1, until=100.0, author='Kishan Sthankiya',
            sim_args_dict=None, logging_interval=None):
    sim = Sim(rng_seed=seed)
    sim_hexgrid_centres, hexgrid_plot = hex_grid_setup(isd=isd, sim_radius=sim_radius)
    for centre in sim_hexgrid_centres[:]:
        x, y = centre
        z = 25.0
        sim.make_cell(interval=until * 1e-2, xyz=[x, y, z], n_subbands=subbands, power_dBm=49.0)
    cell_energy_models_dict = {}
    for cell in sim.cells:
        # FIXME
        cell_energy_models_dict[cell.i] = (CellEnergyModel(cell))
        cell.set_f_callback(cell_energy_models_dict[cell.i].update_cell_power())
    ue_ppp = generate_ppp_points(sim=sim, expected_pts=nues, sim_radius=sim_radius)
    for i in ue_ppp:
        x, y = i
        ue_xyz = x, y, 1.5
        sim.make_UE(xyz=ue_xyz,
                    reporting_interval=until * logging_interval).attach_to_strongest_cell_simple_pathloss_model()
    sim_logger = SimLogger(sim=sim, seed=seed, until=until, sim_args=sim_args_dict)
    energy_logger = EnergyLogger(sim=sim, seed=seed, cell_energy_models=cell_energy_models_dict, until=until,
                                 logging_interval=logging_interval)
    cell_logger = CellLogger(sim=sim, until=until, logging_interval=logging_interval)
    ue_logger = UeLogger(sim=sim, until=until, logging_interval=logging_interval)
    sim.add_loggers([sim_logger, energy_logger, cell_logger, ue_logger])  # std_out & dataframe
    plt.scatter(x=ue_xyz[0], y=ue_xyz[1], s=1.0)
    fig_timestamp(fig=hexgrid_plot, author=author)
    # plt_filepath = QmEnergyLogger.finalize.
    #    today_folder + '/' + filename
    # plt.savefig()
    # plt.show()
    sim.run(until=until)


if __name__ == '__main__':  # a simple self-test
    np.set_printoptions(precision=4, linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=0, help='seed value for random number generator')
    parser.add_argument('-isd', type=float, default=500.0, help='Base station inter-site distance in metres')
    parser.add_argument('-sim_radius', type=float, default=1000.0, help='Simulation bounds radius in metres')
    parser.add_argument('-nues', type=int, default=10, help='number of UEs')
    parser.add_argument('-subbands', type=int, default=1, help='number of subbands')
    parser.add_argument('-until', type=float, default=10.0, help='simulation time')
    parser.add_argument('-logging_interval', type=float, default=1.0,
                        help='Logging interval (in seconds) for the functions that will capture simulation data and for how often the UEs will send reports to their cells.')

    args = parser.parse_args()
    test_01(seed=args.seed, subbands=args.subbands, isd=args.isd, sim_radius=args.sim_radius, nues=args.nues,
            until=args.until, sim_args_dict=args.__dict__, logging_interval=args.logging_interval)
