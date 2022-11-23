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


import argparse
from dataclasses import dataclass
from datetime import datetime
from os import getcwd
from pathlib import Path

import pandas as pd
from numpy import pi
from shapely.geometry import box
from hexalattice.hexalattice import *

from AIMM_simulator import Cell, UE, Scenario, Sim, from_dB, Logger


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
                    avg_se = ec / s.logging_interval / (
                                cell.bw_MHz * 1e6) / s.energy_model.cell_sectors  # Average spectral efficiency (bit/s/Hz/TRxP)
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
        timestamp = datetime.now()
        timestamp_iso = timestamp.isoformat()
        filename = str(Path(__file__).stem + '_log_' + timestamp_iso)
        s.main_dataframe.to_csv(filename, index=False)


# END class QmEnergyLogger

class QmScenario(Scenario):

    # This loop sets the amount of time between each event
    def loop(s, interval=10.0):
        while True:
            yield s.sim.wait(interval)


# END class QmScenario

def create_bbox(length=1000.0): # FIXME need to create a circular bound
    """
    Create a bounding box for the simulation.
    length: float
        Scalar value of the bounding box in metres. Default is 1000.0.
    return: shapely.geometry.polygon.Polygon
        Rectangular polygon with configurable normal vector
    """
    length = float(length)
    return box(minx=0.0, miny=0.0, maxx=length, maxy=length, ccw=False)


def generate_ppp_points(n_pts=100, sim_radius=500.0):
    """
    Generates npts number of points, distributed according to a homogeneous PPP
    with intensity lamb and returns an array of distances to the origin.
    """

    n = 0
    xx = []
    yy = []
    while n < n_pts:
        # Generate the radius value
        radius_polar = sim_radius * np.sqrt(np.random.uniform(0, 1, 1))

        # Generate theta value
        theta = np.random.uniform(0, 2 * pi, 1)

        # Convert to cartesian coords
        x = radius_polar * np.cos(theta)
        y = radius_polar * np.sin(theta)

        # Add to the array
        xx.append(float(x))
        yy.append(float(y))

        n = n + 1

    if n == n_pts:
        return np.array(list(zip(xx, yy)))


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
    ax.set_xlim([-1300, 1300])
    ax.set_ylim([-1300, 1300])
    ax.set_aspect('equal')
    return hexgrid_xy, fig


def test_01(seed=0, isd=500.0, sim_radius=1000, nues=10, until=10.0):
    sim = Sim(rng_seed=seed)
    sim_hexgrid_centres, hexgrid_plot = hex_grid_setup(isd=isd, sim_radius=sim_radius)
    for centre in sim_hexgrid_centres[:]:
        cell_xyz = np.empty(3)
        cell_xyz[:2] = centre
        cell_xyz[2] = 20.0
        sim.make_cell(interval=1.0, xyz=cell_xyz)
    ue_ppp = generate_ppp_points(n_pts=nues, sim_radius=sim_radius)
    for i, xy in enumerate(ue_ppp):
        ue_xyz = np.append(xy, 2.0)
        sim.make_UE(xyz=ue_xyz).attach_to_strongest_cell_simple_pathloss_model()
    em = Energy(sim)
    for cell in sim.cells:
        cell.set_power_dBm(49)
        cell.set_f_callback(em.f_callback, cell_i=cell.i)
    for ue in sim.UEs:
        ue.set_f_callback(em.f_callback, ue_i=ue.i)
    logger = QmEnergyLogger(sim=sim, energy_model=em, logging_interval=1.0)
    sim.add_logger(logger)  # std_out & dataframe
    scenario = QmScenario(sim, verbosity=0)
    sim.add_scenario(scenario)
    plt.scatter(x=ue_ppp[:, 0], y=ue_ppp[:, 1], marker='.', s=10)
    plt.show()
    sim.run(until=until)
    print(f'cell_energy_totals={em.cell_energy_totals}joules')
    print(f'UE_energy_totals  ={em.ue_energy_totals}joules')


if __name__ == '__main__':  # a simple self-test
    np.set_printoptions(precision=4, linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=0, help='seed value for random number generator')
    parser.add_argument('-isd', type=float, default=500.0, help='Base station inter-site distance in metres')
    parser.add_argument('-sim_radius', type=float, default=1000.0, help='Simulation bounds radius in metres')
    parser.add_argument('-nues', type=int, default=10, help='number of UEs')
    parser.add_argument('-until', type=float, default=15.0, help='simulation time')
    args = parser.parse_args()
    test_01(seed=args.seed, isd=args.isd, sim_radius=args.sim_radius, nues=args.nues,
            until=args.until)
