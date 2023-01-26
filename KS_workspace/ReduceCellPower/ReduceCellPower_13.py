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
# v11 - fixed the cell power cons equation
# v11a - Tidy up: 1) plt statements; 2) Comment out unused print statements.
# Testing with AIMM v2.02 to see if clean copy of AIMM simulator yields different results
# Added UE distance from serving cell to UE Logger.
# FIXME - need a better way to output tables for different seed values & power levels.


import argparse
import json
import sys
import progressbar
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
    p_max_dbm: float = 23.0
    power_static_watts: float = 6.8
    eta_pa: float = 0.067
    power_rf_watts: float = 1.0
    power_baseband_watts: float = 3.0
    loss_feed_db: float = 0.00
    loss_dc: float = 0.09
    loss_cool: float = 0.00
    loss_mains: float = 0.11
    delta_p: float = 4.0  # 10.1109/LCOMM.2013.091213.131042
    sectors: int = 1
    antennas: int = 2


@dataclass(frozen=True)
class MacroCellParameters:
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

        # Log the cell id to make sure that only the owner Cell instance can update via a callback function
        s.cell_id = cell.i
        logging.debug("Attaching CellEnergyModel to Cell[%s]", s.cell.i)
        if s.cell.get_power_dBm() >= 30.0:
            logging.debug("Cell[%s] transmit power > 30 dBm.", s.cell.i)

            s.cell_type = 'MACRO'
            logging.debug("Cell[%s] type set to %s.", cell.i, s.cell_type)

            s.params = MacroCellParameters()
            logging.debug("Cell[%s] params set to %s.", cell.i, s.params.__class__.__name__)

        else:
            logging.debug("Cell[%s] transmit power < 30 dBm.", s.cell.i)

            s.cell_type = 'SMALL'
            logging.debug("Cell[%s] type set to %s.", s.cell.i, s.cell_type)

            s.params = SmallCellParameters()
            logging.debug("Cell[%s] params set to %s.", s.cell.i, s.params.__class__.__name__)

        # List of params to store
        s.CELL_POWER_OUT_DBM_MAX = s.params.p_max_dbm
        logging.debug("Cell[%s] P_out_Cell_max_dBm: %s.", s.cell.i, s.CELL_POWER_OUT_DBM_MAX)

        s.SECTOR_TRX_CHAIN_POWER_KILOWATTS_STATIC = s.params.power_static_watts / 1000  # baseline energy use
        logging.debug("Cell[%s] P_out_Sector_TRXchain_static_kW: %s.", s.cell.i,
                      s.SECTOR_TRX_CHAIN_POWER_KILOWATTS_STATIC)

        s.SECTOR_TRX_CHAIN_POWER_KILOWATTS_DYNAMIC = s.trx_chain_power_dynamic_kW()  # The load based power consumption
        logging.debug("Cell[%s] P_out_Sector_TRXchain_dynamic_kW: %s.", s.cell.i,
                      s.SECTOR_TRX_CHAIN_POWER_KILOWATTS_DYNAMIC)

        # Calculate the starting cell power
        s.cell_power_kW = s.params.sectors * s.params.antennas * (
                s.SECTOR_TRX_CHAIN_POWER_KILOWATTS_STATIC + s.SECTOR_TRX_CHAIN_POWER_KILOWATTS_DYNAMIC)
        logging.debug("Starting power consumption for Cell[%s] (kW): %s", s.cell.i, s.cell_power_kW)

        # END of INIT

    def from_dBm_to_watts(s, x):
        """Converts dBm (decibel-milliwatt) input value to watts"""
        return from_dB(x) / 1000

    def get_power_out_per_trx_chain_watts(s, cell_power):
        """
        Takes an input value for a cell power output in dBm.
        Returns the power output of a single TRX chain in Watts.
        A TRX chain consists of an antenna, power amplifier, rf unit and baseband unit.
        """
        return s.from_dBm_to_watts(cell_power)

    def trx_chain_power_dynamic_kW(s):
        """
        Returns the power consumption (in kW), per sector / antenna.
        """
        cell_p_out_dBm = s.cell.get_power_dBm()

        if cell_p_out_dBm > s.CELL_POWER_OUT_DBM_MAX:
            raise ValueError('Power cannot exceed the maximum cell power!')

        # Get current TRX chain output power in watts
        trx_p_out_watts = s.get_power_out_per_trx_chain_watts(cell_p_out_dBm)

        # Sanity check that other input values are in decimal form
        p_rf_watts = s.params.power_rf_watts
        p_bb_watts = s.params.power_baseband_watts

        # Calculate the Power Amplifier power consumption in watts
        p_pa_watts = trx_p_out_watts / (s.params.eta_pa * from_dB(1 - s.params.loss_feed_db))

        # Calculate power consumptions of a single TRX chain (watts)
        p_consumption_watts = p_pa_watts + p_rf_watts + p_bb_watts

        # Calculate losses (ratio)
        p_losses_ratio = (1 - s.params.loss_dc) * (1 - s.params.loss_mains) * (1 - s.params.loss_cool)

        # Get the power output per TRX chain (watts)
        p_out_TRX_chain_watts = p_consumption_watts / p_losses_ratio

        # Power output per TRX chain (kW)
        p_out_TRX_chain_kW = p_out_TRX_chain_watts / 1000

        # Update the instance stored value
        s.SECTOR_TRX_CHAIN_POWER_KILOWATTS_DYNAMIC = p_out_TRX_chain_kW

        return p_out_TRX_chain_kW

    def update_cell_power_kW(s):
        # First update the cell power
        s.cell_power_kW = s.params.sectors * s.params.antennas * (
                s.SECTOR_TRX_CHAIN_POWER_KILOWATTS_STATIC + s.trx_chain_power_dynamic_kW())
        logging.debug('Cell[%s] power consumption has been updated to: %s', s.cell.i, s.cell_power_kW)

    def get_cell_power_kW(s, time):
        if time == 0:
            logging.debug('Cell[%s] power consumption at t=0 is %s', s.cell.i, s.cell_power_kW)
            return s.cell_power_kW
        else:
            s.update_cell_power_kW()
            logging.debug('Cell[%s] power consumption at t=%s is %s', s.cell.i, time, s.cell_power_kW)
            return s.cell_power_kW

    def f_callback(s, x, **kwargs):
        if isinstance(x, Cell):
            if x.i == s.cell_id:
                s.update_cell_power_kW()
            else:
                logging.warning('Cell[%s] is trying to update the Cell[%s] energy model.', x.i, s.cell_id)
                raise ValueError('Cells can only update their own energy model instances! Check the cell_id.')


# End class Energy


class SimLogger(Logger):
    """
       Custom Logger for Simulation config and output.
       """

    def __init__(s, sim, seed, until, sim_args=None, func=None, header='', f=stdout,
                 logging_interval=1.0, experiment_suffix=None, experiment_name=None):
        s.experiment_suffix = experiment_suffix
        s.experiment_name = experiment_name
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
    def log_folder(experiment_name):
        script_parent_dir = Path(__file__).resolve().parents[1]
        logging_path = str(script_parent_dir) + '/logfiles/' + str(Path(__file__).stem)
        if experiment_name is not None:
            today_folder = Path(logging_path + '/' + SimLogger.date_str() + '/' + experiment_name)
        else:
            today_folder = Path(logging_path + '/' + SimLogger.date_str())
        today_folder.mkdir(parents=True, exist_ok=True)
        return str(today_folder)

    @classmethod
    def write_log(cls, logtype_: _LOGTYPES, sim_args_dict: dict = None, df: pd.DataFrame = None,
                  experiment_name: str = None, experiment_suffix: str = None):
        if logtype_ is not None:
            options = get_args(cls._LOGTYPES)
            assert logtype_ in options, f"'{logtype_}' is not in {options}"
            logfile_name = experiment_suffix + '_' + logtype_ + 'Log' + '_' + cls.time_str()
            filepath = SimLogger.log_folder(experiment_name) + '/' + logfile_name
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
        s.write_log(sim_args_dict=s.sim_args, logtype_="Config", experiment_name=s.experiment_name,
                    experiment_suffix=s.experiment_suffix)


# END class SimLogger

class CellLogger(Logger):
    """
    Custom Logger for energy modelling.
    """

    def __init__(s, sim, seed, until, sim_args=None, func=None, header='', f=stdout,
                 logging_interval=1.0, experiment_suffix=None, experiment_name=None):
        s.experiment_suffix = experiment_suffix
        s.experiment_name = experiment_name
        s.until: float = until
        s.seed = seed
        s.cell_dataframe = None  # Create empty placeholder for later pd.DataFrame
        super(CellLogger, s).__init__(sim, func, header, f, logging_interval, np_array_to_str=np_array_to_str)
        s.sim_args = sim_args

    def get_cell_data(s):
        # Create a dictionary of the variables we are interested in
        data = {
            "seed": [s.seed] * s.sim.get_ncells(),
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
            if s.sim.env.now == 0:
                yield s.sim.wait(s.logging_interval)
            # if there isn't a dataframe, initialise it!
            if s.cell_dataframe is None:
                s.initialise_dataframe()

            # if there is already a dataframe, add the cell data to it.
            else:
                s.add_to_cell_dataframe()
            yield s.sim.wait(s.logging_interval)

    def finalize(s):
        # print(s.cell_dataframe)
        SimLogger.write_log(logtype_="Cell", df=s.cell_dataframe, experiment_name=s.experiment_name,
                            experiment_suffix=s.experiment_suffix)


# END class CellLogger


class UeLogger(Logger):
    """
    Custom Logger for UE data.
    """

    def __init__(s, sim, seed, until, sim_args=None, func=None, header='', f=stdout,
                 logging_interval=1.0, experiment_suffix=None, experiment_name=None):
        s.experiment_suffix = experiment_suffix
        s.experiment_name = experiment_name
        s.seed = seed
        s.until: float = until
        s.ue_dataframe = None  # Create empty placeholder for later pd.DataFrame
        super(UeLogger, s).__init__(sim, func, header, f, logging_interval, np_array_to_str=np_array_to_str)
        s.sim_args = sim_args

    def get_interference(s, ue):
        if ue.sinr_dB is None:
            return float(0.0)
        else:
            return float(ue.serving_cell.get_rsrp(ue.i) / ue.sinr_dB)

    def get_distance_serving_cell(s, ue):
        ue_xy = ue.xyz[:2]
        serving_cell_xy = ue.serving_cell.xyz[:2]
        dist = np.linalg.norm(serving_cell_xy - ue_xy)
        return dist

    def get_ue_data(s):
        # Create a dictionary of the variables we are interested in
        data = {
            "seed": [s.seed] * s.sim.get_nues(),
            "time": [s.sim.env.now] * s.sim.get_nues(),
            "ue_id": [k.i for k in s.sim.UEs],
            "ue_xyz": [str(k.xyz).strip('[] ') for k in s.sim.UEs],
            "ue_reporting_interval": [k.reporting_interval for k in s.sim.UEs],
            "ue_serving_cell_id": [k.serving_cell.i for k in s.sim.UEs],
            "ue_serving_cell_rsrp": [k.serving_cell.get_rsrp(k.i) for k in s.sim.UEs],
            "background_thermal_noise_dBm": [k.noise_power_dBm for k in s.sim.UEs],
            "pathloss_model": [k.pathloss.__class__.__name__ for k in s.sim.UEs],
            "pathloss_dBm": [k.pathloss(k.serving_cell.xyz, k.xyz) for k in s.sim.UEs],
            "ue_neighbouring_cell_interference_dBm": [s.get_interference(k) for k in s.sim.UEs],
            "ue_sinr_dB": [k.sinr_dB for k in s.sim.UEs],
            "ue_cqi": [k.serving_cell.get_UE_CQI(k.i).item() for k in s.sim.UEs],
            "ue_pdsch_tput_Mbps": [k.serving_cell.get_UE_throughput(k.i) for k in s.sim.UEs],
            "ue_distance_to_cell": [s.get_distance_serving_cell(k) for k in s.sim.UEs]
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
            if s.sim.env.now == 0:
                yield s.sim.wait(s.logging_interval)

            # if there isn't a dataframe, initialise it!
            if s.ue_dataframe is None:
                s.initialise_dataframe()

            # if there is already a dataframe, add the cell data to it.
            else:
                s.add_to_ue_dataframe()
            yield s.sim.wait(s.logging_interval)

    def finalize(s):
        # print(s.ue_dataframe)
        SimLogger.write_log(logtype_="UE", df=s.ue_dataframe, experiment_name=s.experiment_name,
                            experiment_suffix=s.experiment_suffix)


# END class UeLogger


class EnergyLogger(Logger):
    """
    Custom Logger for energy modelling.
    """

    def __init__(s, sim, seed, cell_energy_models, until, func=None, header='', f=stdout, logging_interval=1.0,
                 experiment_suffix=None, experiment_name=None):
        s.seed = seed
        s.experiment_suffix = experiment_suffix
        s.experiment_name = experiment_name
        s.cell_energy_models = cell_energy_models
        s.until: float = until
        s.energy_dataframe = None  # Placeholder for later pd.DataFrame
        super(EnergyLogger, s).__init__(sim, func, header, f, logging_interval, np_array_to_str=np_array_to_str)

    def get_energy_data(s):
        if s.sim.env.now == 0:
            return

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

        def get_cell_power_kW(cell: Cell):
            """Get the energy model object for this cell"""

            # Get the CellEnergyModel object from the dictionary for this cell
            cell_em = s.cell_energy_models[cell.i]

            # Get current time
            now = s.sim.env.now

            return cell_em.get_cell_power_kW(now)

        def get_cell_ee(cell: Cell):
            """ Get the cell Energy Efficiency (bits/second/watt)"""
            return get_pdsch_throughput_Mbps(cell) / get_cell_power_kW(cell)

        def get_cell_se(cell: Cell):
            """ Get the cell Spectral Efficiency (bits/second/Hertz)"""
            return get_pdsch_throughput_Mbps(cell) / cell.bw_MHz

        # Create a dictionary of the variables we are interested in
        data = {
            "seed": [s.seed] * s.sim.get_ncells(),
            "time": [s.sim.env.now] * s.sim.get_ncells(),
            "cell_id": [k.i for k in s.sim.cells],
            "cell_tx_power_dBm": [k.power_dBm for k in s.sim.cells],
            "cell_nattached_ues": [k.get_nattached() for k in s.sim.cells],
            "pdsch_throughput_Mbps": [get_pdsch_throughput_Mbps(k) for k in s.sim.cells],
            "cell_power_cons_kW": [get_cell_power_kW(k) for k in s.sim.cells],
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
            if s.sim.env.now == 0:
                yield s.sim.wait(s.logging_interval)

            # if there isn't a dataframe, initialise it!
            if s.energy_dataframe is None:
                s.initialise_dataframe()

            # if there is already a dataframe, add the cell data to it.
            else:
                s.add_to_energy_dataframe()
            yield s.sim.wait(s.logging_interval)

    def finalize(s):
        # print(s.energy_dataframe)
        SimLogger.write_log(logtype_="Energy", df=s.energy_dataframe, experiment_name=s.experiment_name,
                            experiment_suffix=s.experiment_suffix)


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


def test_01(seed=0, subbands=1, isd=5000.0, sim_radius=2500.0, nues=1, until=100.0, power_dBm=43.0,
            author='Kishan Sthankiya',
            sim_args_dict=None, logging_interval=None, experiment_name=None):
    # Set up the Simulator instance
    sim = Sim(rng_seed=seed)

    # Create the 19-cell hex-grid and place Cell instance at the centre
    sim_hexgrid_centres, hexgrid_plot = hex_grid_setup(isd=isd, sim_radius=sim_radius)
    for centre in sim_hexgrid_centres[:]:
        x, y = centre
        z = 25.0
        sim.make_cell(interval=until * 1e-2, xyz=[x, y, z], n_subbands=subbands, power_dBm=power_dBm)

    # Create a dictionary of cell-specific energy models
    cell_energy_models_dict = {}
    for cell in sim.cells:
        cell_energy_models_dict[cell.i] = (CellEnergyModel(cell))
        cell.set_f_callback(cell_energy_models_dict[cell.i].f_callback(cell))

    # Generate UEs using PPP and add to simulation
    ue_ppp = generate_ppp_points(sim=sim, expected_pts=nues, sim_radius=sim_radius)
    for i in ue_ppp:
        x, y = i
        ue_xyz = x, y, 1.5
        sim.make_UE(xyz=ue_xyz,
                    reporting_interval=until * logging_interval).attach_to_strongest_cell_simple_pathloss_model()

    # Set experiment suffix
    experiment_suffix = str(f"s{seed}_t{until:.0f}_nues{nues:.0f}_dBm{power_dBm:.0f}")

    # Set up loggers and add to simulation
    sim_logger = SimLogger(sim=sim, seed=seed, until=until, sim_args=sim_args_dict, experiment_name=experiment_name,
                           experiment_suffix=experiment_suffix)
    energy_logger = EnergyLogger(sim=sim, seed=seed, cell_energy_models=cell_energy_models_dict, until=until,
                                 logging_interval=logging_interval, experiment_name=experiment_name,
                                 experiment_suffix=experiment_suffix)
    cell_logger = CellLogger(sim=sim, until=until, seed=seed, logging_interval=logging_interval,
                             experiment_name=experiment_name,
                             experiment_suffix=experiment_suffix)
    ue_logger = UeLogger(sim=sim, until=until, seed=seed, logging_interval=logging_interval,
                         experiment_name=experiment_name,
                         experiment_suffix=experiment_suffix)
    sim.add_loggers([sim_logger, energy_logger, cell_logger, ue_logger])  # std_out & dataframe

    # Plot setup if desired
    # plt.scatter(x=ue_xyz[0], y=ue_xyz[1], s=1.0)
    fig_timestamp(fig=hexgrid_plot, author=author)

    # Run the simulation
    sim.run(until=until)


if __name__ == '__main__':  # a simple self-test
    np.set_printoptions(precision=4, linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=0, help='seed value for random number generator')
    parser.add_argument('-isd', type=float, default=500.0, help='Base station inter-site distance in metres')
    parser.add_argument('-sim_radius', type=float, default=1000.0, help='Simulation bounds radius in metres')
    parser.add_argument('-nues', type=int, default=10, help='number of UEs')
    parser.add_argument('-subbands', type=int, default=1, help='number of subbands')
    parser.add_argument('-power_dBm', type=float, default=43.0, help='set the transmit power of the cell in dBm')
    parser.add_argument('-until', type=float, default=2.0, help='simulation time')
    parser.add_argument('-logging_interval', type=float, default=1.0,
                        help='Sampling interval (seconds) for simulation data capture + UEs reports sending.')
    parser.add_argument('-experiment_name', type=str, default='Exp01',
                        help='name of a specific experiment to influence the output log names.')

    args = parser.parse_args()
    test_01(seed=args.seed, subbands=args.subbands, isd=args.isd, sim_radius=args.sim_radius, power_dBm=args.power_dBm,
            nues=args.nues,
            until=args.until, sim_args_dict=args.__dict__, logging_interval=args.logging_interval,
            experiment_name=args.experiment_name)
