# Kishan Sthankiya
# Basic energy model class based on test_f_callback_00.py
# Uses callback functions to compute the energy.
# Added dataclasses for small cells and macro cells.
# Added bounding box with shapely.geometry.
# 2022-11-20 13:55 Add custom QmEnergyLogger
# Add cell power_dBm level to logger
# Add hexgrid functions!!!
# Add average spectral efficiency to logger


import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from os import getcwd
from pathlib import Path
from time import strftime

import numpy as np
import pandas as pd
from AIMM_simulator import Cell, UE, Scenario, Sim, from_dB, Logger
from matplotlib import pyplot as plt
from scipy.spatial import voronoi_plot_2d, Voronoi
from shapely.geometry import box


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
    p_max_db: float = 43.0
    power_static_watts: float = 130.0
    eta_pa: float = 0.311
    power_rf_watts: float = 12.9
    power_baseband_watts: float = 29.6
    loss_feed_db: float = -3.0
    loss_dc_db: float = 0.075
    loss_cool_db: float = 0.10
    loss_mains_db: float = 0.09


class Energy:
    """
    Defines a complete self-contained system energy model.
    """

    def __init__(s, sim):
        """ Initialize variables which will accumulate energy totals.
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

    def cell_energy(s, cell):
        """
          Increment cell power consumption for one simulation timestep.
          Based on EARTH framework (10.1109/MWC.2011.6056691).
        """
        if cell.get_power_dBm() < 30:
            trx = s.params_small_cell
            s.cell_power_static = s.params_small_cell.power_static_watts
        else:
            trx = s.params_macro_cell
            s.cell_power_static = s.params_macro_cell.power_static_watts

        if cell.pattern is None:
            cell_antennas = 3  # Assume 3*120 degree antennas
        else:
            cell_antennas = 1  # If an array or function, assume it is unidirectional (for now)
        s.cell_sectors = cell_antennas  # Assuming 3 sectors. FIX when complex antennas implemented.

        n_trx = cell.n_subbands * cell_antennas * s.cell_sectors  # Number of transceiver chains
        trx_power_max = from_dB(trx.p_max_db)  # The maximum transmit power in watts
        trx_power_now = from_dB(cell.get_power_dBm())  # The current transmit power in watts

        def trx_power(p):
            """
            Calculates the power consumption for a given transmit power level (in watts), per transceiver.
            """
            if 0.0 < p < trx_power_max:
                trx_power_pa = p / trx.eta_pa * (1 - from_dB(trx.loss_feed_db))  # Power amplifier in watts
                trx_power_sum = trx_power_pa + trx.power_rf_watts + trx.power_baseband_watts
                trx_power_losses = (1 - trx.loss_dc_db) * (1 - trx.loss_mains_db) * (1 - trx.loss_cool_db)
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
            # print(f't={s.sim.env.now:.1f}: cell[{x.i}] (check from kwargs: {kwargs["cell_i"]}) energy={s.cell_energy_totals[x.i]:.0f}kW')
            s.cell_energy(x)
        elif isinstance(x, UE):
            s.ue_power(x)


class QmEnergyLogger(Logger):
    """
    Custom Logger for energy modelling.
    """

    def __init__(s, energy_model, *args, **kwargs):
        Logger.__init__(s, *args, **kwargs)
        s.energy_model: Energy = energy_model
        s.cols = (
            'time',
            'cell',
            'cell_dBm',
            'n_UEs',
            'tp_bps',
            'Energy(J)',
            'EE',
            'SE'
        )
        s.ec = np.zeros(s.sim.get_ncells())
        s.main_dataframe = pd.DataFrame(data=None, columns=list(s.cols))  # Create empty pd.DataFrame with headings

    def append_row(s, new_row):
        temp_df = pd.DataFrame(data=[new_row], columns=list(s.cols))
        s.main_dataframe = pd.concat([s.main_dataframe, temp_df])

    def loop(s):
        # Write to stdout
        yield s.sim.wait(s.logging_interval)
        s.f.write("#time\tcell\tcell_dBm\tn_ues\ttp_bps\tEnergy (J)\tEE (bits/J)\tavg_SE(bit/s/Hz/TRxP)\n")
        while True:
            # Needs to be per cell in the simulator
            for cell in s.sim.cells:
                tm = s.sim.env.now  # timestamp
                cell_dbm = cell.get_power_dBm()
                n_ues = cell.get_nattached()  # attached UEs
                tp = 0.0  # total throughput set to ZERO
                tp = sum(cell.get_UE_throughput(ue_i) for ue_i in cell.attached)  # Update throughput
                tp_bits = tp * 1e+6  # Convert throughput from Mbps>bps
                ec = s.energy_model.cell_energy_now[cell.i]
                s.ec += ec
                if ec == 0.0:  # Calculate the energy efficiency
                    ee = 0.0
                    avg_se = 0.0
                else:
                    ee = tp_bits / ec
                    avg_se = ec / s.logging_interval / (cell.bw_MHz * 1e6) / s.energy_model.cell_sectors  # Average spectral efficiency (bit/s/Hz/TRxP)
                # Write to stdout
                s.f.write(
                    f"{tm:10.2f}\t{cell.i:2}\t{cell_dbm:2}\t{n_ues:2}\t{tp_bits:.2e}\t{ec:.2e}\t{ee:10.2f}\t{avg_se:.2e}\n"
                )
                # Write these variables to the main_dataframe
                row = (tm, cell.i, cell_dbm, n_ues, tp_bits, ec, ee, avg_se)
                new_row = [np.round(each_arr, decimals=2) for each_arr in row]
                s.append_row(new_row)

            yield s.sim.wait(s.logging_interval)

    def finalize(s):
        cwd = getcwd()
        filename = str(Path(__file__).stem + '_log_' + strftime("%Y%m%d-%H%M%S"))
        s.main_dataframe.to_csv(filename, index=False)


# END class QmEnergyLogger

def create_bbox(length=1000.0):
    """
    Create a bounding box for the simulation.
    length: float
        Scalar value of the bounding box in metres. Default is 1000.0.
    return: shapely.geometry.polygon.Polygon
        Rectangular polygon with configurable normal vector
    """
    length = float(length)
    return box(minx=0.0, miny=0.0, maxx=length, maxy=length, ccw=False)


def ppp(sim, n_ues, x_max, y_max, x_min=0, y_min=0):
    n_points = sim.rng.poisson(n_ues)
    x = (x_max * np.random.uniform(0, 1, n_points) + x_min)  # FIX: not a tuple
    y = (y_max * np.random.uniform(0, 1, n_points) + y_min)  # FIX: not a tuple
    return np.stack((x, y), axis=1)


def plot_ppp(ue_arr, ax_x_max, ax_y_max):
    plt.scatter(x=ue_arr[:, 0], y=ue_arr[:, 1], marker='.', color='r')
    plt.xlim(0, ax_x_max)
    plt.ylim(0, ax_y_max)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot()
    plt.show()


def create_hexgrid(bbox, side):
    """
    Returns an array of Points describing hexagons centers that are inside the given bounding_box.
    param bbox: The containing bounding box. The bbox coordinate should be in Webmercator
    param side: The size of the hexagons
    return: The hexagon grid
    """
    grid = []

    v_step = math.sqrt(3) * side
    h_step = 1.5 * side

    x_min = min(bbox[0], bbox[2])
    x_max = max(bbox[0], bbox[2])
    y_min = min(bbox[1], bbox[3])
    y_max = max(bbox[1], bbox[3])

    h_skip = math.ceil(x_min / h_step) - 1
    h_start = h_skip * h_step

    v_skip = math.ceil(y_min / v_step) - 1
    v_start = v_skip * v_step

    h_end = x_max + h_step
    v_end = y_max + v_step

    if v_start - (v_step / 2.0) < y_min:
        v_start_array = [v_start + (v_step / 2.0), v_start]
    else:
        v_start_array = [v_start - (v_step / 2.0), v_start]

    v_start_idx = int(abs(h_skip) % 2)

    c_x = h_start
    c_y = v_start_array[v_start_idx]
    v_start_idx = (v_start_idx + 1) % 2
    while c_x < h_end:
        while c_y < v_end:
            grid.append((c_x, c_y))
            c_y += v_step
        c_x += h_step
        c_y = v_start_array[v_start_idx]
        v_start_idx = (v_start_idx + 1) % 2

    return grid


def make_hexgrid(bbox, xlim, ylim, side_length):
    sim_bbox = bbox
    sim_bbox_vertices = list(sim_bbox.bounds)
    hex_grid = create_hexgrid(sim_bbox_vertices, side_length)
    points = np.array(hex_grid)
    vor = Voronoi(points=points)

    # Plot the hex grid and axis view
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='grey',
                    line_width=0.6, line_alpha=0.2, point_size=5)
    ax.set_aspect('equal')
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    return points, ax.plot()


def test_01(seed=0, boxlength=100.0, sidelength=150, ncells=1, nues=1, until=10.0):
    sim = Sim(rng_seed=seed)
    sim_box = create_bbox(boxlength)
    sim_box_xmin, sim_box_ymin, sim_box_xmax, sim_box_ymax = sim_box.bounds
    points, hex_plot = make_hexgrid(bbox=sim_box, xlim=sim_box_xmax, ylim=sim_box_ymax, side_length=sidelength)
    for i in range(ncells):
        cell_xyz = np.empty(3)
        cell_xyz[:2] = sim_box_xmin + sim_box_xmax * sim.rng.random(2)
        cell_xyz[2] = 20.0
        sim.make_cell(interval=1.0, xyz=cell_xyz)
    ue_ppp = ppp(sim=sim, n_ues=nues, x_max=sim_box_xmax, y_max=sim_box_ymax)
    for i, xy in enumerate(ue_ppp):
        ue_xyz = np.append(xy, 2.0)
        sim.make_UE(xyz=ue_xyz).attach_to_strongest_cell_simple_pathloss_model()
    em = Energy(sim)
    for cell in sim.cells:
        cell.set_f_callback(em.f_callback, cell_i=cell.i)
    for ue in sim.UEs:
        ue.set_f_callback(em.f_callback, ue_i=ue.i)
    plot_ppp(ue_arr=ue_ppp, ax_x_max=sim_box_xmax, ax_y_max=sim_box_ymax)
    scenario = Scenario(sim, verbosity=0)
    sim.add_scenario(scenario)
    logger = QmEnergyLogger(sim=sim, energy_model=em, logging_interval=1.0)
    sim.add_logger(logger)  # std_out & dataframe
    sim.run(until=until)
    print(f'cell_energy_totals={em.cell_energy_totals}joules')
    print(f'UE_energy_totals  ={em.ue_energy_totals}joules')


if __name__ == '__main__':  # a simple self-test
    np.set_printoptions(precision=4, linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=0, help='seed value for random number generator')
    parser.add_argument('-boxlength', type=float, default=1000.0, help='simulation bounding box length in metres')
    parser.add_argument('-sidelength', type=float, default=150, help='hexagon side length in metres')
    parser.add_argument('-ncells', type=int, default=4, help='number of cells')
    parser.add_argument('-nues', type=int, default=10, help='number of UEs')
    parser.add_argument('-until', type=float, default=15.0, help='simulation time')
    args = parser.parse_args()
    test_01(seed=args.seed, boxlength=args.boxlength, sidelength=args.sidelength, ncells=args.ncells, nues=args.nues,
            until=args.until)
