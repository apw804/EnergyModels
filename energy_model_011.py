# Kishan Sthankiya
# 2022-11-15 10:00:12
# Restructure of code with different working parts
# Dataframe needs Jinja2. Install with pip or conda.

import math
from dataclasses import dataclass
from datetime import datetime
from os import getcwd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from AIMM_simulator import Cell, UE, Logger, Scenario, Sim, from_dB, Radio_state, to_dB
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, box

rng = np.random.default_rng(seed=0)


#  PPP
# -----

# Poisson Point Process: UE position as xy
def ppp(sim, n_ues, x_max, y_max, x_min=0, y_min=0):
    n_points = sim.rng.poisson(n_ues)
    x = (x_max * np.random.uniform(0, 1, n_points) + x_min)  # FIX: not a tuple
    y = (y_max * np.random.uniform(0, 1, n_points) + y_min)  # FIX: not a tuple
    return np.stack((x, y), axis=1)


# Plot the points for all UEs (for illustration only)
def plot_ppp(ppp_arr, ax_x_max, ax_y_max):
    plt.scatter(x=ppp_arr[:, 0], y=ppp_arr[:, 1], edgecolor='b', facecolor='none', alpha=0.5)
    plt.xlim(0, ax_x_max)
    plt.ylim(0, ax_y_max)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot()
    plt.show()


#  Hexagons
# ----------

def create_hexagon(length, x, y):
    """
    Create a hexagon centered on (x, y)
    :param length: length of the hexagon's edge
    :param x: x-coordinate of the hexagon's center
    :param y: y-coordinate of the hexagon's center
    :return: The polygon containing the hexagon's coordinates
    """
    c = [[x + math.cos(math.radians(angle)) * length, y + math.sin(math.radians(angle)) * length] for angle in
         range(0, 360, 60)]
    return Polygon(c)


def create_bbox(length=None):
    if length is None:
        length = 1
    else:
        length = float(length)
        return box(minx=0.0, miny=0.0, maxx=length, maxy=length, ccw=False)


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


def test_hexgrid(bbox=15):
    bbox_size = bbox
    sim_bbox = create_bbox(bbox_size)
    sim_bbox_vertices = list(sim_bbox.bounds)
    hex_grid = create_hexgrid(sim_bbox_vertices, 1)
    points = np.array(hex_grid)
    vor = Voronoi(points=points)

    # Plot the hex grid and axis view
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='grey',
                    line_width=0.6, line_alpha=0.2, point_size=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.plot()
    plt.show()


@dataclass(frozen=True)
class SmallCellParameters:
    """ Object for setting small cell base station parameters."""
    p_max_db: float = 23.0
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
    eta_pa: float = 0.067
    power_rf_watts: float = 12.9
    power_baseband_watts: float = 29.6
    loss_feed_db: float = -3.0
    loss_dc_db: float = 0.075
    loss_cool_db: float = 0.10
    loss_mains_db: float = 0.09


class EarthModel:
    """
    Power model for a base station.
    Based on EARTH framework (10.1109/MWC.2011.6056691).
    """

    def __init__(self, cell):
        self.cell = cell
        self.bs_power_watts = from_dB(self.cell.get_power_dBm)
        if self.cell.get_power_dBm < 30:
            self.cell.params = SmallCellParameters()
        else:
            self.cell.params = MacroCellParameters()
        # Store the maximum output power for the base station type
        self.power_bs_max_watts = from_dB(self.cell.params.p_max_db)

        if self.cell.pattern is None:
            self.cell.antennas = 3  # Assume base station is 3 antennas of 120 degrees
        else:
            if self.cell.pattern == np.array:
                self.cell.antennas = 1  # If there's an array assume that it's 1 antenna
            else:
                self.cell.antennas = 1  # assuming the function is unidirectional (for now)
        self.cell.sectors = self.cell.antennas
        self.cell.loss_feeder_ratio = from_dB(self.cell.params.loss_feed_db)
        self.cell_power_totals = np.zeros(sim.get_ncells())
        self.ue_a_kW = 1.0e-3  # slope
        self.ue_energy_totals = np.zeros(sim.get_nues())
        self.p_out = self.bs_power_watts

    def rsp_dbm(self):
        """
        Calculates the Reference Signal Power in dBm. Based on maximum number of physical resource blocks in the
        current radio state.
        """
        max_rbs = Radio_state.nPRB
        return self.cell.get_power_dBm - to_dB(max_rbs * 12)  # 12 is the number of resource elements in a PRB

    def power_tx_per_ue_dbm(self):
        """
        Calculates transmit dBm per UE.
        Evenly divides available transmit power over the attached UEs
        """
        available_tx_dbm = self.p_out - self.rsp_dbm()
        attached_ues = self.cell.get_nattached()
        return available_tx_dbm / attached_ues

    def power_pa_watts(self, p_out):
        return p_out / (self.cell.params.eta_pa * (1 - self.cell.loss_feeder_ratio))

    def n_trx(self):
        return self.cell.n_subbands * self.cell.antennas * self.cell.sectors

    def power_bs_sum_watts(self, p_out):
        return self.power_pa_watts(p_out) + self.cell.params.power_rf_watts + self.cell.params.power_baseband_watts

    def power_bs_loss(self):
        return (1 - self.cell.params.loss_dc_db) * (1 - self.cell.params.loss_mains_db) * (
                    1 - self.cell.params.loss_cool_db)

    def power_bs_per_trx(self, p_out):
        return self.power_bs_sum_watts(p_out) / self.power_bs_loss()

    # This is for when there is ZERO load on the base station
    def power_bs_static(self, p_out=0.0):
        return self.power_bs_per_trx(p_out)

    # This is for when there is any NON-ZERO load on the base station (up to max load).
    def power_bs_dynamic(self, p_out):
        if 0.0 < p_out < self.power_bs_max_watts:
            delta_p = self.power_bs_max_watts / self.power_bs_static()
            return self.power_bs_static() + (delta_p * p_out)

    def power_bs_active(self, p_out):
        static = self.power_bs_static()
        dynamic = self.power_bs_dynamic(p_out)
        return self.n_trx() * (static + dynamic)

    def power_bs_sleep(self):
        return self.n_trx() * self.power_bs_static()

    def power_bs_cons(self, p_out):
        return self.power_bs_static() + self.power_bs_dynamic(p_out)

    def cell_power(self, cell):
        """
          Increment cell energy usage for one simulation timestep.
        """
        self.cell_power_totals[cell.i] += cell.interval * self.power_bs_cons(p_out=self.p_out)

    def get_cell_total_power(self, cell):
        """
          Return current total energy usage for cell.
        """
        return self.cell_power_totals[cell.i]

    def ue_energy(self, ue):
        """
          Increment UE energy usage for one simulation timestep.
        """
        self.ue_energy_totals[ue.i] += ue.reporting_interval * self.ue_a_kW



# END class EarthModel

class QmEnergyLogger(Logger):
    """
    Custom Logger for the EarthModel energy model.
    """

    # Constructor for QmEnergyLogger with additional parameter
    def __init__(s, *args, **kwargs):
        # Calling the parent constructor using ClassName.__init__()
        Logger.__init__(s, *args, **kwargs)
        # Adding the cols and main_dataframe
        s.cols = (
            'time',
            'cell',
            'n_UEs',
            'tp_Mbps',
            'Power(W)',
            'EE',
        )
        s.main_dataframe = pd.DataFrame(data=None, columns=list(s.cols))

    def append_row(s, new_row):
        temp_df = pd.DataFrame(data=[new_row], columns=list(s.cols))
        s.main_dataframe = pd.concat([s.main_dataframe, temp_df])

    def loop(s):
        # Write to stdout
        yield s.sim.wait(s.logging_interval)
        s.f.write("#time\tcell\tn_ues\ttp_Mbps\tPower (W)\tEE (Mbps/kW)\n")
        while True:
            # Needs to be per cell in the simulator
            for cell in s.sim.cells:
                tm = s.sim.env.now  # timestamp
                n_ues = cell.get_nattached()    # attached UEs
                tp = 0.0                        # total throughput set to ZERO
                tp = sum(cell.get_UE_throughput(ue_i) for ue_i in cell.attached)    # Update throughput
                pc = EarthModel(s.sim, cell).get_cell_total_power(cell)   # Power consumption for cell
                # Calculate the energy efficiency
                if pc == 0.0:
                    ee = 0.0  # KB think about types - should this be 0.0?
                else:
                    ee = tp / pc
                # Write to stdout
                s.f.write(
                    f"{tm:10.2f}\t{cell.i:2}\t{n_ues:2}\t{tp:10.2f}\t{pc:10.2f}\t{ee:10.2f}\n"
                )
                # Write these variables to the main_dataframe
                row = (tm, cell.i, n_ues, tp, pc, ee)
                new_row = [np.round(each_arr, decimals=2) for each_arr in row]
                s.append_row(new_row)

            yield s.sim.wait(s.logging_interval)

    def plot(s):

        # Separate out into dataframes for each cell[i]
        df = s.main_dataframe
        df_cell0 = df[df.cell.eq(0)]
        df_cell1 = df[df.cell.eq(1)]
        df_cell2 = df[df.cell.eq(2)]

        # Cell0 plot
        # x-axis is time
        x = df_cell0["time"]
        y_0_tp = df_cell0["tp_Mbps"]
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
        fig.savefig('foo.png')
        # plt.show()
        # print(s.main_dataframe)

    def finalize(s):
        cwd = getcwd()
        filename = str(Path(__file__).stem + '_log_' + str(datetime.now(tz=None)))
        f = open(filename, 'w')
        # x_formatted=f'{x:6g}'.ljust(7,'0') # 6 significant figures, left-justified and right-padded with zeros
        # f'{x_formatted}'
        s.main_dataframe.style.format('{>0.2f}') # Need to install Jinja2
        s.main_dataframe.to_csv(f, sep='\t', index=False, encoding='ascii')
        # f.close()
        s.plot()


# END class QmEnergyLogger

class QmScenario(Scenario):

    # This loop sets the amount of time between each event
    def loop(s, interval=0.1):
        while True:
            yield s.sim.wait(interval)


# END class QmScenario

def test(seed=1, ncells=1, nues=1, plane_x_max=1000, plane_y_max=1000, until=1000.0):
    """
    Define the parameters to set up and execute the simulation.
    """
    interval = 1.0e0  # KB cell interval - other intervals will be scaled to this
    sim = Sim(rng_seed=seed)
    # Setup cells
    for i in range(ncells):
        sim.make_cell(interval=interval, verbosity=1)
    # Setup UEs
    ue_ppp = ppp(sim=sim, n_ues=nues, x_max=plane_x_max, y_max=plane_y_max)
    plot_ppp(ppp_arr=ue_ppp, ax_x_max=plane_x_max, ax_y_max=plane_y_max)
    for i, xy in enumerate(ue_ppp):
        ue_xyz = np.append(xy, 2.0)
        ue_reporting_interval = 0.1 * interval  # UE intervals are scaled to a higher rate than the sim
        sim.make_UE(xyz=ue_xyz, reporting_interval=ue_reporting_interval).attach_to_nearest_cell()
    
    # Add logger & scenario
    sim.add_logger(QmEnergyLogger(sim, logging_interval=1 * interval))      # std_out & dataframe
    scenario = QmScenario(sim, verbosity=1)                                 # Does nothing - UEs are stationary
    sim.add_scenario(scenario)  # FIXME interval?
    sim.run(until=until)



if __name__=='__main__': # a simple self-test
  np.set_printoptions(precision=6,linewidth=200)
  parser=argparse.ArgumentParser()
  parser.add_argument('-seed',type=int,default= 1,help='random number generator seed')
  parser.add_argument('-ncells',type=int,default= 4,help='number of cells')
  parser.add_argument('-nues',type=int,default=10,help='number of UEs')
  parser.add_argument('-plane_x_max',type=int,default=1000,help='size of plane x-coords')
  parser.add_argument('-plane_y_max',type=int,default=1000,help='size of plane y-coords')
  parser.add_argument('-until',type=float,default=100.0,help='simulation time')
  args=parser.parse_args()
  test(seed=args.seed,ncells=args.ncells,nues=args.nues,plane_x_max=args.plane_x_max,plane_y_max=args.plane_y_max,until=args.until)
