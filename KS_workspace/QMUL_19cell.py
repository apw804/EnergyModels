# Simple script with 19 cells. 0 UEs.
import argparse
from dataclasses import dataclass
from collections import namedtuple
import numpy as np
from AIMM_simulator import Cell, Logger, Sim
from hexalattice.hexalattice import *


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


class QmCellLogger(Logger):
    """
    Basic custom Logger for Cell data points
    """

    def get_cell_data(s):
        n_cells = s.sim.get_ncells()
        time = [s.sim.env.now()]
        labels = []
        data = []
        cell_id = cell_i.i,
        cell_xyz = cell_i.get_xyz(),
        cell_subbands = cell_i.get_subband_mask(),
        n_attached_ues = cell_i.get_nattached(),
        power_dBm = cell_i.get_power_dBm(),
        rsrp = cell_i.get_rsrp(),
        avg_pdsch_tput_Mbps = cell_i.get_average_throughput()
        col_labels = [i for i, j in locals().items()]
        cell_data.extend([time, cell_id, cell_xyz, cell_subbands, n_attached_ues, power_dBm, rsrp, avg_pdsch_tput_Mbps])
        return data

    def create_dataframe(s, data):
        pass

    def loop(s, custom_logging_interval=1):
        '''
        Main loop of Logger class.
        Can be overridden to provide custom functionality.
        '''
        while True:
            s.get_cell_data()
            yield s.sim.env.timeout(custom_logging_interval)

    def finalize(s):
        '''
        Function called at end of simulation, to implement any required finalization actions.
        '''
        pass
    def loop(s):
        while True:
            s.get_cell_data()
        yield s.logging_interval()


def hex_grid_setup(origin: tuple = (0, 0), isd: float = 500.0, sim_radius: float = 1000.0):
    """
    Creates a hex grid of 19 site (57 sectors) using the hexalattice module when using the defaults.
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
    circle_dashed = plt.Circle(
        origin, sim_radius, fill=False, linestyle='--', color='r')

    ax.add_patch(circle_dashed)
    ax.scatter(hexgrid_x, hexgrid_y, marker='2')
    # Factor to set the x,y-axis limits relative to the isd value.
    ax_scaling = 2 * isd + 500
    ax.set_xlim([-ax_scaling, ax_scaling])
    ax.set_ylim([-ax_scaling, ax_scaling])
    ax.set_aspect('equal')
    return hexgrid_xy, fig


def test_01(seed=0, subbands=1, isd=1.0, sim_radius=2.50, until=1.0, sim_args=None):
    sim = Sim(rng_seed=seed)
    sim_hexgrid_centres, hexgrid_plot = hex_grid_setup(
        isd=isd, sim_radius=sim_radius)
    for centre in sim_hexgrid_centres[:]:
        cell_xyz = np.empty(3)
        cell_xyz[:2] = centre
        cell_xyz[2] = 20.0
        sim.make_cell(interval=1.0, xyz=cell_xyz,
                      n_subbands=subbands, power_dBm=43)
    cell_logger = QmCellLogger(sim=sim)
    sim.add_logger(cell_logger)  # std_out & dataframe
    sim.run(until=until)


if __name__ == '__main__':
    test_01()
