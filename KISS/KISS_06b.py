# KISS (Keep It Simple Stupid!) v3
# Now gets the dataframe output as floats instead of np.ndarrays
# FIXME - need to finish integrating the NR_5G_standard... max cell throughput
# Scenario: Runs Richard's suggestion to reduce the cell power of the outer ring of cells.
# FIXME - add the AMF to limit the UEs that are attached to Cells based on whether they have a CQI>0
# Added the updated MCS table2 from PHY.py-data-procedures to override NR_5G_standard_functions.MCS_to_Qm_table_64QAM
# Adding sleep mode by subclassing Cell and Sim
# Refactored components from ReduceCellPower_14.py: CellEnergyModel


import argparse, json
import logging
from pathlib import Path
from sys import stdout, stderr
from datetime import datetime
from time import localtime, strftime
from types import NoneType
from attr import dataclass
import numpy as np
from AIMM_simulator import *
from hexalattice.hexalattice import *
import matplotlib.pyplot as plt
import pandas as pd
from _PHY import phy_data_procedures

logging.basicConfig(stream=stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
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


class Cellv2(Cell):
    """ Class to extend original Cell class and add functionality"""

    _SLEEP_MODES = [0,1,2,3,4]

    def __init__(self, *args, energy_model=None, sleep_mode=None, **kwargs):   # [How to use *args and **kwargs with 'peeling']
        # Set Cellv2 energy_model
        if isinstance(energy_model, NoneType):
            self.energy_model = CellEnergyModel(cell=self)
        if isinstance(sleep_mode, NoneType):
            self.sleep_mode = 0
        else:
            self.sleep_mode = sleep_mode
        # print(f'Cell[{self.i}] sleep mode is: {self.sleep_mode}')
        super().__init__(*args, **kwargs)

    def set_mcs_table(self, mcs_table_number):
        """
        Changes the lookup table used by NR_5G_standard_functions.MCS_to_Qm_table_64QAM
        """
        if mcs_table_number == 1:
            NR_5G_standard_functions.MCS_to_Qm_table_64QAM = phy_data_procedures.mcs_table_1     # same as LTE
        elif mcs_table_number == 2:
            NR_5G_standard_functions.MCS_to_Qm_table_64QAM = phy_data_procedures.mcs_table_2     # 5G NR; 256QAM
        elif mcs_table_number == 3:
            NR_5G_standard_functions.MCS_to_Qm_table_64QAM = phy_data_procedures.mcs_table_3     # 5G NR; 64QAM LowSE/RedCap (e.g. IoT devices)
        elif mcs_table_number == 4:
            NR_5G_standard_functions.MCS_to_Qm_table_64QAM = phy_data_procedures.mcs_table_4     # 5G NR; 1024QAM
        # print(f'Setting Cell[{self.i}] MCS table to: table-{mcs_table_number}')
        return
    
    def set_sleep_mode(self, mode:int):
        """
         Sets the Cell sleep mode. If set to a number between 1-4, changes behaviour of Cell and associated energy consumption.
         Default: self.sleep_mode = 0 (NO SLEEP MODE) - Cell is ACTIVE
        """
        if mode not in self._SLEEP_MODES:
            raise ValueError("Invalid sleep mode. Must be one of the following: %r." % self._SLEEP_MODES)

        # Uncomment below to debug
        # print(f'Changing Cell[{self.i}] sleep mode from {self.sleep_mode} to {mode}')
        self.sleep_mode = mode

        # If the simulation is initialised but not running yet, wait until it is running.
        if self.sim.env.now < 1:
            yield self.sim.env.timeout(self.interval)

        # If the cell is in REAL sleep state (1-4):
        if self.sleep_mode in self._SLEEP_MODES[1:]:  
            orphaned_ues = self.sim.orphaned_ues

            # DEBUG message
            print(f'Cell[{self.i}] is in SLEEP_MODE_{self.sleep_mode}')

            # Cellv2.power_dBm should be -inf
            self.power_dBm = -np.inf

            # ALL attached UE's should be detached (orphaned)
            for i in self.attached:
                ue = self.sim.UEs[i]
                orphaned_ues.append(i)
                ue.detach                           # This should also take care of UE throughputs.
            
            # Orphaned UEs should attach to cells with best RSRP
            self.sim.mme.attach_ues_to_best_rsrp(orphaned_ues)

    
    def get_sleep_mode(self):
        """
        Return the sleep_mode for the Cellv2.
        """
        return self.sleep_mode

    
    def loop(self):
        '''
        Main loop of Cellv2 class.
        Default: Checks if the sleep_mode flag is set and adjusts cell behaviour accordingly.
        '''
        while True:
            self.get_sleep_mode()
            if self.f_callback is not None: self.f_callback(self,**self.f_callback_kwargs)
            yield self.sim.env.timeout(self.interval)
    

class Simv2(Sim):
    """ Class to extend original Sim class for extended capabilities from sub-classing."""
    def __init__(self, *args, **kwargs):
        self.orphaned_ues = []
        super().__init__(*args, **kwargs)
    
    def make_cellv2(self, **kwargs):
            ''' 
            Convenience function: make a new Cellv2 instance and add it to the simulation; parameters as for the Cell class. Return the new Cellv2 instance.).
            '''
            self.cells.append(Cellv2(self,**kwargs))
            xyz=self.cells[-1].get_xyz()
            self.cell_locations=np.vstack([self.cell_locations,xyz])
            return self.cells[-1]


# Calculate the maximum Resource Elements
max_nRE = NR_5G_standard_functions.Radio_state.nPRB * NR_5G_standard_functions.Radio_state.NRB_sc
    




class AMFv1(MME):
    """
    Adds to the basic AIMM MME and rebrands to the 5G nomenclature AMF(Access and Mobility Management Function).
    """

    def __init__(self, *args, cqi_limit:int = None, **kwargs):
        self.cqi_limit = cqi_limit
        self.poor_cqi_ues = []
        super().__init__(*args, **kwargs)

    def check_low_cqi_ue(self, ue_ids, threshold=None):
        """
        Takes a list of UE IDs and adds the UE ID to `self.poor_cqi_ues` list.
        """
        self.poor_cqi_ues.clear()
        threshold = self.cqi_limit

        for i in ue_ids:
            ue = self.sim.UEs[i]

            if isinstance(threshold, NoneType):
                return
            if isinstance(ue.cqi, NoneType):
                return
            if ue.cqi[-1] < threshold:
                self.poor_cqi_ues.append(ue.i)
                return 

    def detach_low_cqi_ue(self, poor_cqi_ues=None):
        """
        Takes a list self.poor_cqi_ues (IDs) and detaches from their serving cell.
        """
        if isinstance(poor_cqi_ues, NoneType):
            poor_cqi_ues = self.poor_cqi_ues

        for ue_id in poor_cqi_ues:
            ue = self.sim.UEs[ue_id]
            # Add the UE to the `sim.orphaned_ues` list.
            self.sim.orphaned_ues.append(ue_id)
            # Finally, detach the UE from it's serving cell.
            ue.detach()
        
        # Finally, clear the `self.poor_cqi_ues` list
        self.poor_cqi_ues.clear()


    def attach_ues_to_best_rsrp(self, ues):
        """
        Accepts a list of UEs. Attaches UE to the cell that gives it the best rsrp.
        """
        for ue_i in ues:
            ue = self.sim.UEs[ue_i]
            celli=self.sim.get_best_rsrp_cell(ue_i)
            ue.attach(self.sim.cells[celli])
            ue.send_rsrp_reports() # make sure we have reports immediately
            ue.send_subband_cqi_report()


    def best_sinr_cell():
           # TODO - write function to replace the 'best_rsrp_cell' strategy.
        pass

    def loop(self):
        '''
        Main loop of AMFv1.
        '''
        while self.sim.env.now < 1:    # At t=0 there will not be handover events
            yield self.sim.env.timeout(0.5*self.interval)   # So we stagger the MME startup by 0.5*interval
        print(f'MME started at {float(self.sim.env.now):.2f}, using strategy="{self.strategy}" and anti_pingpong={self.anti_pingpong:.0f}.',file=stderr)
        while True:
            self.do_handovers()
            self.detach_low_cqi_ue()
            yield self.sim.env.timeout(self.interval)
    
    def finalize(self):
        self.detach_low_cqi_ue()
        super().finalize()


class ChangeCellPower(Scenario):
    """
    Changes the power_dBm of the specified list of cells (default outer ring) after a specified delay time (if provided), relative to t=0.
    """

    def __init__(self,sim,interval=0.5, cells=None, delay=None, new_power=None):
        self.target_cells = cells
        self.delay_time = delay
        self.new_power = new_power
        self.sim = sim
        self.interval = interval
        self.outer_ring = [0,1,2,3,6,7,11,12,15,16,17,18]
        if cells is None:
            self.target_cells = self.outer_ring


    def loop(self):
        while True:
            if self.sim.env.now < self.delay_time:
                yield self.sim.wait(self.interval)
            if self.sim.env.now > self.delay_time:
                for i in self.target_cells:
                    self.sim.cells[i].set_power_dBm(self.new_power)
            yield self.sim.wait(self.interval)

class MyLogger(Logger):

    def __init__(self, *args, **kwargs):
        self.dataframe = None
        super().__init__(*args, **kwargs)

    def get_cqi_to_mcs(self, cqi):
        """
        Returns the MCS value for a given CQI value. Copied from `NR_5G_standard_functions.CQI_to_64QAM_efficiency`
        """
        if type(cqi) is NoneType:
            return np.nan
        else:
            return max(0,min(28,int(28*cqi/15.0)))

    def get_max_5G_throughput(self, mcs):       # FIXME - incomplete
        """
        Returns the max 5G throughput for a given MCS - from `NR_5G_standard_functions.py`
        """
        if  type(mcs) is int:
            # Update the mcs and nPRB of the radio_state dataclass
            current_radio_state = NR_5G_standard_functions.Radio_state
            current_radio_state.MCS = mcs
            current_radio_state.nPRB = 24   # FIXME - will need to come from lookup tables

            # Get value for _DMRS_RE?

            # Run the max throughput function
            max_tp = NR_5G_standard_functions.max_5G_throughput_64QAM(radio_state=current_radio_state)

            return max_tp
        
        else:
            return np.nan

    def get_neighbour_cell_rsrp_rank(self, ue_id, neighbour_rank):
        neighbour_cell_rsrp = []
        for cell in self.sim.cells:
            cell_id = cell.i
            ue_rsrp =  cell.get_RSRP_reports_dict()[ue_id]      # get current UE rsrp from neighbouring cells
            neighbour_cell_rsrp += [[cell_id, ue_rsrp]]
        neighbour_cell_rsrp.sort(key=lambda x: x[1], reverse=True)
        neighbour_cell_rsrp.pop(0)
        neighbour_id, neighbour_rsrp_dBm = neighbour_cell_rsrp[neighbour_rank]
        return neighbour_id, neighbour_rsrp_dBm
    
    def get_cell_power_kW(self, cell: Cellv2):
            """Get the energy model object for this cell"""

            # Get the CellEnergyModel object from the dictionary for this cell
            cell_em = self.cell_energy_models[cell.i]

            # Get current time
            now = self.sim.env.now

            return cell_em.get_cell_power_kW(now)


    def get_data(self):
        # Create an empty list to store generated data
        data = []
        # Keep a list of column names to track
        columns = ["time", "serving_cell_id", "serving_cell_sleep_mode", "ue_id",
            "distance_to_cell(m)", "throughput(Mb/s)", "sc_power(dBm)","sc_rsrp(dBm)", "neighbour1_rsrp(dBm)", "neighbour2_rsrp(dBm)", "noise_power(dBm)", "sinr(dB)", "cqi", "mcs"]
        for cell in self.sim.cells:
               for attached_ue_id in cell.attached:
                    UE = self.sim.UEs[attached_ue_id]
                    serving_cell = UE.serving_cell
                    tm = self.sim.env.now                                           # current time
                    sc_id = serving_cell.i                                          # current UE serving_cell
                    sc_sleep_mode = serving_cell.get_sleep_mode()                   # current UE serving_cell sleep mode status
                    sc_xy = serving_cell.get_xyz()[:2]                              # current UE serving_cell xy position
                    ue_id = UE.i                                                    # current UE ID
                    ue_xy = UE.get_xyz()[:2]                                        # current UE xy position
                    d2sc = np.linalg.norm(sc_xy - ue_xy)                            # current UE distance to serving_cell
                    tp = serving_cell.get_UE_throughput(attached_ue_id)             # current UE throughput ('fundamental')
                    sc_power = serving_cell.get_power_dBm()                         # current UE serving_cell transmit power
                    sc_rsrp = serving_cell.get_RSRP_reports_dict()[ue_id]           # current UE rsrp from serving_cell
                    neigh1_rsrp = self.get_neighbour_cell_rsrp_rank(ue_id, 0)[1]    # current UE neighbouring cell 1 rsrp
                    neigh2_rsrp = self.get_neighbour_cell_rsrp_rank(ue_id, 1)[1]    # current UE neighbouring cell 2 rsrp
                    noise = UE.noise_power_dBm                                      # current UE thermal noise
                    sinr = UE.sinr_dB                                               # current UE sinr from serving_cell
                    cqi = UE.cqi                                                    # current UE cqi from serving_cell
                    mcs = self.get_cqi_to_mcs(cqi)                                  # current UE mcs for serving_cell
                    # tp_max = self.get_max_5G_throughput(mcs)                      # current UE max throughput

                    # Get the above as a list
                    data_list = [tm, sc_id, sc_sleep_mode, ue_id, d2sc, tp, sc_power, sc_rsrp, neigh1_rsrp, neigh2_rsrp, noise, sinr, cqi, mcs]

                    # convert ndarrays to str or float
                    for i, j in enumerate(data_list):
                        if type(j) is np.ndarray:
                            data_list[i] = float(j)

                    # Write above to `data` list
                    data.append(data_list)
        return columns, data

    def add_to_dataframe(self, col_labels, new_data, ignore_index):
        if self.dataframe is None:
            self.dataframe = pd.DataFrame(new_data, columns=col_labels)
        else:
            new_data_df = pd.DataFrame(data=new_data, columns=col_labels)
            self.dataframe = pd.concat(
                [self.dataframe, new_data_df], verify_integrity=True, ignore_index=ignore_index)

    def run_routine(self, ignore_index=True):
        col_labels, new_data = self.get_data()
        self.add_to_dataframe(col_labels=col_labels,
                              new_data=new_data, ignore_index=ignore_index)

    def loop(self):
        """Main loop for logger."""
        while True:
            # Don't capture t=0
            if self.sim.env.now == 0:
                yield self.sim.wait(self.logging_interval)
            self.run_routine()
            print(f'logger time={self.sim.env.now}')
            yield self.sim.wait(self.logging_interval)

    def finalize(self):
        '''
        Function called at end of simulation, to implement any required finalization actions.
        '''
        print(f'Finalize time={self.sim.env.now}')

        # Run routine for final time step
        self.run_routine(ignore_index=True)

        # Create a copy of the final DataFrame
        df = self.dataframe.copy()

        # Sort the data by time, UE_id then cell_id
        df1 = df.sort_values(['time','ue_id', 'serving_cell_id'], ascending=[True, True, True])

        # Print df to screen
        print(df)

        # (DEBUGGING TOOL) 
        # Print a view of the type of value in each position
        # --------------------------------------------------
        # df_value_type = df.applymap(lambda x: type(x).__name__)
        # print(df_value_type)

        # Write the MyLogger dataframe to TSV file
        df.to_csv(logfile, sep="\t", index=False)

# END MyLogger class


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
    # intensity (ie mean density) of the Poisson process
    lambda0 = expected_pts / areaTotal

    # Simulate Poisson point process
    # Poisson number of points
    numbPoints = sim_rng.poisson(lambda0 * areaTotal)
    # angular coordinates
    theta = 2 * np.pi * sim_rng.uniform(0, 1, numbPoints)
    # radial coordinates
    rho = sim_radius * np.sqrt(sim_rng.uniform(0, 1, numbPoints))

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
    circle_dashed = plt.Circle(
        origin, sim_radius, fill=False, linestyle='--', color='r')

    ax.add_patch(circle_dashed)
    ax.scatter(hexgrid_x, hexgrid_y, marker='2')
    # Factor to set the x,y-axis limits relative to the isd value.
    ax_scaling = 3 * isd + 500
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


def write_args_to_json(outfile, parse_args_obj):
       if not outfile.endswith('.json'):
            outfile += '.json'
            # convert namespace to dictionary
            args_dict = vars(parse_args_obj)
            # write to json file
            with open(outfile, 'w') as f:
                json.dump(args_dict, f, indent=4)

def plot_ues(sim, ue_ids: list):
    ue_objs_list = [sim.UEs[i] for i in ue_ids]
    ue_x_list = [ue.xyz[0] for ue in ue_objs_list]
    ue_y_list = [ue.xyz[1] for ue in ue_objs_list]
    ue_xy_list = [ue.xyz[:2] for ue in ue_objs_list]
    plt.scatter(x=ue_x_list, y=ue_y_list, color='red', s=2.0)
    for i in ue_ids:
        plt.annotate(text=str(i), xy=ue_xy_list[i], xytext=(3,-2), textcoords='offset points',
        fontsize=8, color='red', bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),)

def main(seed, isd, sim_radius, power_dBm, nues, until, base_interval, new_power_dBm, mcs_table_number, author=None):
    # Create a simulator object
    sim = Simv2(rng_seed=seed)

    # Scale other intervals to base_interval
    ue_reporting_interval = base_interval
    logging_interval = base_interval

    # Create instance of UMa-NLOS pathloss model
    pl_uma_nlos = UMa_pathloss(LOS=False)

    # Create the 19-cell hex-grid and place Cell instance at the centre
    sim_hexgrid_centres, hexgrid_plot = hex_grid_setup(isd=isd, sim_radius=sim_radius)
    for centre in sim_hexgrid_centres[:]:
        x, y = centre
        z = 25.0
        # Create the cell
        sim.make_cellv2(interval=base_interval*0.5,
                      xyz=[x, y, z], power_dBm=power_dBm)

 
    for cell in sim.cells:
        # Set the MCS table for the Cell
        cell.set_mcs_table(mcs_table_number=mcs_table_number)

        # Quick and simple labelling of cell_ids
        cell_id = cell.i
        cell_x = cell.xyz[0]
        cell_y = cell.xyz[1]
        plt.annotate(cell_id, (cell_x, cell_y), color='blue', alpha=0.3)

    # Generate UE positions using PPP
    ue_ppp = generate_ppp_points(sim=sim, expected_pts=nues, sim_radius=sim_radius)
    for i in ue_ppp:
        x, y = i
        ue_xyz = x, y, 1.5
        sim.make_UE(xyz=ue_xyz, reporting_interval=ue_reporting_interval, pathloss_model=pl_uma_nlos).attach_to_strongest_cell_simple_pathloss_model()

    # Change the noise_power_dBm for all UEs to -118dBm
    for ue in sim.UEs:
        ue.noise_power_dBm=-118.0

    # Add the logger to the simulator
    custom_logger = MyLogger(sim, logging_interval=logging_interval)
    sim.add_logger(custom_logger)

    # Add scenario to simulation
    change_outer_ring_power = ChangeCellPower(sim, delay=0, new_power=new_power_dBm, interval=base_interval)
    sim.add_scenario(scenario=change_outer_ring_power)

    # Add MME for handovers
    default_mme = AMFv1(sim, cqi_limit=1, interval=base_interval,strategy='best_rsrp_cell',anti_pingpong=0.0,verbosity=2)
    sim.add_MME(mme=default_mme)

    # Plot UEs if desired (uncomment to activate)
    sim_ue_ids = [ue.i for ue in sim.UEs]
    plot_ues(sim=sim, ue_ids=sim_ue_ids)
    fig_timestamp(fig=hexgrid_plot, author='Kishan Sthankiya')
    fig_outfile_path = Path(logfile).with_suffix('.png')
    plt.savefig(fig_outfile_path)

    # Try setting sleep mode
    sim.cells[2].set_sleep_mode(mode=1)    # testing SLEEP_MODE

    # Run simulator
    sim.run(until=until)


if __name__ == '__main__':  # run the main script

    # Create cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-seeds', type=int, default=0, help='seed value for random number generator')
    parser.add_argument('-isd', type=float, default=1500.0, help='Base station inter-site distance in metres')
    parser.add_argument('-sim_radius', type=float, default=3000.0, help='Simulation bounds radius in metres')
    parser.add_argument('-mcs_table_number', type=int, default=2, help='Set the MCS table for the cell lookup')
    parser.add_argument('-power_dBm', type=float, default=30.0, help='Cell transmit power in dBm.')
    parser.add_argument('-new_power_dBm', type=float, default=21.0, help='Updated cell transmit power in dBm.')
    parser.add_argument('-nues', type=int, default=10, help='number of UEs')
    parser.add_argument('-until', type=float, default=2.0,  help='simulation time')
    parser.add_argument('-base_interval', type=float, default=1.0,  help='base interval for simulation steps')

    # Create the args namespace
    args = parser.parse_args()

    # Set variables for later use
    script_name = Path(__file__).stem
    timestamp = datetime.now()
    timestamp_iso = timestamp.isoformat(timespec='seconds')
    logfile = '_'.join([script_name, 'logfile', timestamp_iso])+'.tsv'
    # FIXME - move these elsewhere!

    # Write input arguments to file for reference (uncomment to activate)
    # outfile = '_'.join([script_name,'config',timestamp_iso])
    # write_args_to_json(outfile=outfile, parse_args_obj=args)

    # Run the __main__
    main(seed=args.seeds, isd=args.isd, sim_radius=args.sim_radius, mcs_table_number=args.mcs_table_number, power_dBm=args.power_dBm, new_power_dBm=args.new_power_dBm, nues=args.nues,
         until=args.until, base_interval=args.base_interval)
