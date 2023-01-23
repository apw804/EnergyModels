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


import argparse
import json
from dataclasses import dataclass
from os import getcwd
from pathlib import Path
from sys import stdout
from time import strftime, localtime
from typing import Literal, get_args

import pandas as pd
from AIMM_simulator import Cell, UE, Scenario, Sim, from_dB, Logger, np_array_to_str
from hexalattice.hexalattice import *
import numpy as np


@dataclass(frozen=True)
class SmallCellParameters:
    """ Object for setting small cell base station parameters."""
    p_max_db: float = 23.0
    power_static_watts: float = 6.8
    eta_PA: float = 0.067
    power_RF_watts: float = 1.0
    power_baseband_watts: float = 3.0
    loss_feed_dB: float = 0.00
    loss_DC_dB: float = 0.09
    loss_cool_dB: float = 0.00
    loss_mains_dB: float = 0.11


@dataclass(frozen=True)
class MacroCellParameters:
    """ Object for setting macro cell base station parameters."""
    p_max_db: float = 49.0
    power_static_watts: float = 130.0
    eta_pa: float = 0.311
    power_rf_watts: float = 12.9
    power_baseband_watts: float = 29.6
    loss_feed_db: float = -3.0
    loss_dc_db: float = 0.075
    loss_cool_db: float = 0.10
    loss_mains_db: float = 0.09


# noinspection PyMethodParameters
class Energy:
    """
    Defines a complete self-contained system energy model for a 5G base station (gNB).

    Parameters
    ----------
    sim : Sim
        Simulator instance which will manage this energy model.
    """

    def __init__(s, sim):
        """
        Initialize variables which will accumulate energy totals.
        """

        s.sim = sim  # reference to the entire simulation!
        s.params_small_cell = SmallCellParameters()
        s.params_macro_cell = MacroCellParameters()
        s.cell_sectors = None
        s.cell_power_static = None  # baseline energy use
        s.cell_a_kW = 1.0  # slope
        s.ue_a_kW = 1.0e-3  # slope
        s.cell_energy_now = np.zeros(sim.get_ncells())
        s.cell_energy_totals = np.zeros(sim.get_ncells())
        s.ue_energy_now = np.zeros(sim.get_nues())
        s.ue_energy_totals = np.zeros(sim.get_nues())
        s.trx = None
        s.cell_antennas = 0

    def cell_energy(s, cell):
        """
          Increment cell power consumption for one simulation timestep.
          Based on EARTH framework (10.1109/MWC.2011.6056691).
        """

        if s.sim.env.now <= 0:
            if cell.get_power_dBm() < 30:
                s.trx = s.params_small_cell
                s.cell_power_static = s.params_small_cell.power_static_watts
            else:
                s.trx = s.params_macro_cell
                s.cell_power_static = s.params_macro_cell.power_static_watts
            if not isinstance(cell.pattern, list):
                s.cell_antennas = 3  # Assume 3*120 degree antennas
            else:
                s.cell_antennas = 1  # If an array or function, assume it is unidirectional (for now)
            s.cell_sectors = s.cell_antennas  # Assuming 3 sectors. FIX when complex antennas implemented.

        n_trx = cell.n_subbands * s.cell_antennas * s.cell_sectors  # Number of transceiver chains
        trx_power_max = from_dB(s.trx.p_max_db)  # The maximum transmit power in watts
        trx_power_now = from_dB(cell.get_power_dBm())  # The current transmit power in watts

        def trx_power(p):
            """
            Calculates the power consumption for a given transmit power level (in watts), per transceiver.
            """
            if 0.0 <= p <= trx_power_max:
                trx_power_pa = p / s.trx.eta_pa * (1 - from_dB(s.trx.loss_feed_db))  # Power amplifier in watts
                trx_power_sum = trx_power_pa + s.trx.power_rf_watts + s.trx.power_baseband_watts
                trx_power_losses = (1 - s.trx.loss_dc_db) * (1 - s.trx.loss_mains_db) * (1 - s.trx.loss_cool_db)
                return trx_power_sum / trx_power_losses
            if p > trx_power_max:
                raise ValueError('Power cannot exceed the maximum transceiver power!')
            if p < 0.0:
                raise ValueError('Power cannot be below ZERO!')

        s.cell_energy_now[cell.i] = cell.interval * (s.cell_power_static + n_trx * trx_power(p=trx_power_now))
        s.cell_energy_totals[cell.i] += cell.interval * (s.cell_power_static + n_trx * trx_power(p=trx_power_now))

    def ue_power(s, ue):
        """
          Increment UE energy usage for one simulation timestep.
        """
        s.ue_energy_now[ue.i] = ue.reporting_interval * s.ue_a_kW
        s.ue_energy_totals[ue.i] += ue.reporting_interval * s.ue_a_kW

    def f_callback(s, x, **kwargs):
        # print(kwargs)
        if isinstance(x, Cell):
            # print(f't={s.sim.env.now:.1f}: cell[{x.i}] (check from kwargs: {kwargs["cell_i"]})
            # energy={s.cell_energy_totals[x.i]:.0f}kW')
            s.cell_energy(x)
        elif isinstance(x, UE):
            s.ue_power(x)


# noinspection PyMethodParameters
class QmEnergyLogger(Logger):
    """
    Custom Logger for energy modelling.
    """

    def __init__(s, sim, seed, energy_model, until, sim_args=None, func=None, header='', f=stdout,
                 logging_interval=1.0):

        s.energy_model: Energy = energy_model
        s.until: float = until
        s.seed: int = seed
        s.cols = (
            'seed',
            'time',
            'cell_id',
            'cell_xyz',
            'cell_bw_MHz',
            'subbands',
            'MIMO_gain',
            'cell_dBm',
            'n_UEs',
            'tp_bits',
            'tp_avg_bits',
            'cell_ec_now',
            'cell_ee_now',
            'cell_avg_se_now',
            'cell_tp_agg_bits_cum',
            'cell_tp_avg_bits_cum',
            'cell_ec_cum',
            'cell_ee_cum',
            'cell_avg_se_overall'
        )

        s.cell_ec_cum = 0.0
        s.cell_ee = 0.0
        s.cell_tp_agg_bits_cum = 0.0
        s.cell_tp_avg_bits_cum = 0.0
        s.cell_ee_cum = 0.0
        s.cell_avg_se_overall = 0.0
        s.main_dataframe = pd.DataFrame(data=None, columns=list(s.cols))  # Create empty pd.DataFrame with headings
        super(QmEnergyLogger, s).__init__(sim, func, header, f, logging_interval, np_array_to_str=np_array_to_str)
        s.sim_args = sim_args

    def append_row(s, new_row):
        temp_df = pd.DataFrame(data=[new_row], columns=list(s.cols))
        s.main_dataframe = pd.concat([s.main_dataframe, temp_df])

    # noinspection PyPep8Naming
    def loop(s):
        # Write to stdout
        yield s.sim.wait(s.logging_interval)
        while True:
            # Needs to be per cell in the simulator
            for cell in s.sim.cells:
                tm = s.sim.env.now  # timestamp
                cell_id = cell.i
                cell_xyz = cell.get_xyz()
                bw_MHz = cell.bw_MHz
                subbands = cell.n_subbands
                # noinspection PyPep8Naming
                MIMO_gain = cell.MIMO_gain_dB
                pattern = cell.pattern
                cell_dbm = cell.get_power_dBm()
                n_ues = cell.get_nattached()  # attached UEs
                tp = cell.get_average_throughput()  # there is only one UE so avg should also work
                tp_bits = tp * 1e+6  # Convert throughput from Mbps>bps
                tp_avg = cell.get_average_throughput()
                tp_avg_bits = tp_avg * 1e+6  # to bits
                cell_ec_now = s.energy_model.cell_energy_now[cell.i]
                if s.sim.env.now > 0:
                    s.cell_ec_cum += cell_ec_now
                    s.cell_tp_agg_bits_cum += tp_bits
                    s.cell_tp_avg_bits_cum += tp_avg_bits

                if cell_ec_now == 0.0:  # Calculate the energy efficiency
                    cell_ee_now = 0.0
                    cell_avg_se_now = 0.0
                else:
                    cell_ee_now = tp_bits / cell_ec_now
                    # Average spectral efficiency (bit/s/Hz/TRxP)
                    cell_avg_se_now = tp_bits / s.logging_interval / (
                            cell.bw_MHz * 1e6) / s.energy_model.cell_sectors
                    s.cell_ee_cum = s.cell_ee_cum + cell_ee_now
                    s.cell_avg_se_overall = (s.cell_avg_se_overall + cell_avg_se_now) / s.sim.until

                # Write these variables to the main_dataframe
                row = (s.seed, tm, cell_id, cell_xyz, bw_MHz, subbands, MIMO_gain, cell_dbm, n_ues,
                       tp_bits, tp_avg_bits, cell_ec_now, cell_ee_now, cell_avg_se_now,
                       s.cell_tp_agg_bits_cum, s.cell_tp_avg_bits_cum, s.cell_ec_cum, s.cell_ee_cum,
                       s.cell_avg_se_overall)
                new_row = np.asarray(row, dtype=object)
                s.append_row(new_row)

            yield s.sim.wait(s.logging_interval)

    def write_sim_args_file(s, filepath, sim_args):
        output_path = filepath + '_sim_params.txt'  # dont need
        replacement = 'python 3 -m ' + str(Path(__file__).resolve()) + ' '  # don't need
        args_rtrim = sim_args.rstrip(sim_args[-1])
        args_lrtrim = args_rtrim.split('(')[1].split(',')
        args_str = [i.replace(' ', '-') for i in args_lrtrim]
        output_str = ' '.join(args_str)
        output_str_full = replacement + '-' + output_str
        with open(output_path, 'w') as f:
            f.write(output_str_full)

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
        today_folder = Path(logging_path + '/' + QmEnergyLogger.date_str())
        today_folder.mkdir(parents=True, exist_ok=True)
        return str(today_folder)

    _LOGTYPES = Literal["Sim", "Args", "Profile"]

    @staticmethod
    def get_logfile_name(logtype_: _LOGTYPES = None):
        if logtype_ is not None:
            options = get_args(QmEnergyLogger._LOGTYPES)
            assert logtype_ in options, f"'{logtype_}' is not in {options}"
            return str(Path(__file__).stem) + '_' + logtype_ + 'Log_' + QmEnergyLogger.time_str()
        raise Exception(f"A logfile type MUST be specified. E.g. {get_args(QmEnergyLogger._LOGTYPES)}")

    def write_sim_log(s):
        filepath = Path(QmEnergyLogger.log_folder() + '/' + QmEnergyLogger.get_logfile_name(logtype_="Sim"))
        s.main_dataframe.to_csv(str(filepath) + '.tsv', index=False, sep='\t', na_rep='NaN', header=True,
                                float_format='%g')

    def write_args_log(s, sim_args_dict):
        filepath = Path(QmEnergyLogger.log_folder() + '/' + QmEnergyLogger.get_logfile_name(logtype_="Args") + ".json")
        with open(filepath, 'w') as f:
            json.dump(sim_args_dict, f)

    def write_profile_log(s):       # FIXME - Placeholder function to write Python profile output to.
        pass

    def finalize(s):
        s.write_sim_log()
        s.write_args_log(sim_args_dict=s.sim_args)


# END class QmEnergyLogger

class QmScenarioReduceCellPower(Scenario):

    def delta_cell_power(s, cell, p_target: float = 0.0):
        """
        Increase or decrease cell power towards the target power in equal amounts over the
        simulation run time remaining.
        """
        delta_p = p_target - cell.get_power_dBm()
        events_remaining = (s.sim.until - s.sim.env.now) / s.interval
        delta_p_event = delta_p / events_remaining
        new_cell_power = int(cell.get_power_dBm() + delta_p_event)
        cell.set_power_dBm(new_cell_power)

    # This loop sets the amount of time between each event
    def loop(s):
        while True:
            for cell in s.sim.cells:
                s.delta_cell_power(cell)
            yield s.sim.wait(s.interval)


# END class QmScenarioReduceCellPower


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
    Creates a hex grid of 1 site (3 sectors) using the hexalattice module when using the defaults.
    """
    fig, ax = plt.subplots()

    hexgrid_xy, _ = create_hex_grid(nx=1,
                                    ny=1,
                                    min_diam=isd,
                                    # crop_circ=sim_radius,
                                    align_to_origin=True,
                                    edge_color=[0.75, 0.75, 0.75],
                                    h_ax=ax,
                                    do_plot=True)

    hexgrid_x = hexgrid_xy[:, 0]
    hexgrid_y = hexgrid_xy[:, 1]
    # circle_dashed = plt.Circle(origin, sim_radius, fill=False, linestyle='--', color='r')

    # ax.add_patch(circle_dashed)
    ax.scatter(hexgrid_x, hexgrid_y, marker='2')
    ax_scaling = isd + 500  # Factor to set the x,y-axis limits relative to the isd value.
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
            sim_args_dict=None):
    sim = Sim(rng_seed=seed, params={'profile': 'SimLogProfile'})
    sim_hexgrid_centres, hexgrid_plot = hex_grid_setup(isd=isd, sim_radius=sim_radius)
    for centre in sim_hexgrid_centres[:]:
        cell_xyz = np.empty(3)
        cell_xyz[:2] = centre
        cell_xyz[2] = 20.0
        sim.make_cell(interval=1.0, xyz=cell_xyz, n_subbands=subbands)
    #    ue_ppp = generate_ppp_points(sim=sim, expected_pts=nues, sim_radius=sim_radius)
    ue_xyz = 2000.0, 0.0, 2.0
    sim.make_UE(xyz=ue_xyz, reporting_interval=until * 1e-3).attach_to_strongest_cell_simple_pathloss_model()
    em = Energy(sim)
    for cell in sim.cells:
        cell.set_power_dBm(49)
        cell.set_f_callback(em.f_callback, cell_i=cell.i)
    for ue in sim.UEs:
        ue.set_f_callback(em.f_callback, ue_i=ue.i)
    logger = QmEnergyLogger(sim=sim, seed=seed, energy_model=em, until=until, sim_args=sim_args_dict)
    sim.add_logger(logger)  # std_out & dataframe
    scenario = QmScenarioReduceCellPower(sim, verbosity=0)
    sim.add_scenario(scenario)
    plt.scatter(x=ue_xyz[0], y=ue_xyz[1], s=1.0)
    fig_timestamp(fig=hexgrid_plot, author=author)
    # plt_filepath = QmEnergyLogger.finalize.
    #    today_folder + '/' + filename
    # plt.savefig()
    sim.run(until=until)
    print(f'cell_energy_totals={em.cell_energy_totals}joules')
    print(f'UE_energy_totals  ={em.ue_energy_totals}joules')


if __name__ == '__main__':  # a simple self-test
    np.set_printoptions(precision=4, linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=0, help='seed value for random number generator')
    parser.add_argument('-isd', type=float, default=5000.0, help='Base station inter-site distance in metres')
    parser.add_argument('-sim_radius', type=float, default=5000.0, help='Simulation bounds radius in metres')
    parser.add_argument('-nues', type=int, default=1, help='number of UEs')
    parser.add_argument('-subbands', type=int, default=1, help='number of subbands')
    parser.add_argument('-until', type=float, default=100.0, help='simulation time')

    args = parser.parse_args()
    test_01(seed=args.seed, subbands=args.subbands, isd=args.isd, sim_radius=args.sim_radius, nues=args.nues,
            until=args.until, sim_args_dict=args.__dict__)
