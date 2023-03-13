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
import numpy as np
from pathlib import Path
from sys import stdout, stderr
from datetime import datetime
import sys
from time import localtime, strftime
from types import NoneType
from attr import dataclass
import debugpy
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

    def __init__(self, *args, sleep_mode=None, **kwargs):   # [How to use *args and **kwargs with 'peeling']
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

    def __init__(self, sim, interval=0.5, cells=None, delay=None, new_power=None):
        """
        Initializes an instance of the ChangeCellPower class.

        Parameters:
        -----------
        sim : SimPy.Environment
            The simulation environment object.
        interval : float, optional
            The time interval between each power change. Default is 0.5.
        cells : list of int, optional
            The list of cell indices to change power. Default is outer_ring.
        delay : float, optional
            The delay time before changing the cell powers. Default is None.
        new_power : float, optional
            The new power_dBm to set for the specified cells. Default is None.

        Returns:
        --------
        None.
        """
        self.target_cells = cells
        self.delay_time = delay
        self.new_power = new_power
        self.sim = sim
        self.interval = interval
        self.outer_ring = [0, 1, 2, 3, 6, 7, 11, 12, 15, 16, 17, 18]
        if cells is None:
            self.target_cells = self.outer_ring

    def loop(self):
        """
        The main loop of the scenario, which changes the power of specified cells after a delay.

        Parameters:
        -----------
        None.

        Returns:
        --------
        None.
        """
        while True:
            if self.sim.env.now < self.delay_time:
                yield self.sim.wait(self.interval)
            if self.sim.env.now > self.delay_time:
                for i in self.target_cells:
                    self.sim.cells[i].set_power_dBm(self.new_power)
            yield self.sim.wait(self.interval)


class SetCellSleep(Scenario):
    """
    A scenario that sets the sleep level of specified cells at specified times.

    Parameters:
    -----------
    sim : Simulation
        The simulation environment to which the scenario is added.
    interval : float, optional
        The interval at which the loop runs. Default is 0.5.
    time_cell_sleep_level_duration : list of dicts, optional
        A list of dictionaries specifying the time, cell, sleep_level, and duration for which cells will be put to sleep.
        If the sleep duration is -1, the cell will sleep indefinitely. Default is None.
    delay : float, optional
        The delay time before the scenario begins. Default is None.

    Attributes:
    -----------
    sleep_delay_time : float or None
        The delay time before the scenario begins.
    sim : Simulation
        The simulation environment to which the scenario is added.
    interval : float
        The interval at which the loop runs.
    bedtime_stories : list of dicts or None
        A list of dictionaries specifying the time, cell, sleep_level, and duration for which cells will be put to sleep.
        None if time_cell_sleep_level_duration is None.

    Raises:
    -------
    TypeError:
        If time_cell_sleep_level_duration is not a list or an element in it is not a dictionary.
        If the type of the time, cell, sleep_level, or duration values in the dictionary is incorrect.
    KeyError:
        If the dictionary in time_cell_sleep_level_duration does not have keys "time", "cell", "sleep_level", or "duration".
        If the cell in the dictionary is not found in the simulation environment.
    ValueError:
        If the sleep_level value in the dictionary is not between 0 and 4.

    """
    def __init__(self, sim, interval=0.5, time_cell_sleep_level_duration: list = None, delay=None):
        self.sleep_delay_time = delay
        self.sim = sim
        self.interval = interval
        if time_cell_sleep_level_duration is None:
            print(f'WARNING: No cells specified for SetCellSleep. No cells will be put to sleep.')
        else:
            if type(time_cell_sleep_level_duration) is not list:
                raise TypeError(f'Expected list for time_cell_sleep_level_duration, got {type(time_cell_sleep_level_duration)}')
            if len(time_cell_sleep_level_duration) == 0:
                print(f'WARNING: No cells specified for SetCellSleep. No cells will be put to sleep.')
            else:
                for i in time_cell_sleep_level_duration:
                    if not isinstance(i, dict):
                        raise TypeError(f'Expected dict for time_cell_sleep_level_duration, got {type(i)}')
                    if 'time' not in i.keys():
                        raise KeyError(f'Expected key "time" in time_cell_sleep_level_duration, got {i.keys()}')
                    if 'cell' not in i.keys():
                        raise KeyError(f'Expected key "cell" in time_cell_sleep_level_duration, got {i.keys()}')
                    if 'sleep_level' not in i.keys():
                        raise KeyError(f'Expected key "sleep_level" in time_cell_sleep_level_duration, got {i.keys()}')
                    if 'duration' not in i.keys():
                        raise KeyError(f'Expected key "duration" in time_cell_sleep_level_duration, got {i.keys()}')
                    if type(i['time']) is not int and type(i['time']) is not float:
                        raise TypeError(f'Expected int or float for time, got {type(i["time"])}')
                    if type(i['cell']) is not int:
                        raise TypeError(f'Expected int for cell, got {type(i["cell"])}')
                    if 'cell' not in self.sim.cells:
                        raise KeyError(f'Cell {i["cell"]} not found in simulation environment.')
                    if type(i['sleep_level']) is not int:
                        raise TypeError(f'Expected int for sleep_level, got {type(i["sleep_level"])}')
                    if i['sleep_level'] < 0 or i['sleep_level'] > 4:
                        raise ValueError(f'Expected sleep_level to be between 0 and 4, got {i["sleep_level"]}')
                    if type(i['duration']) is not int and type(i['duration']) is not float:
                        raise TypeError(f'Expected int or float for duration, got {type(i["duration"])}')
                self.bedtime_stories = time_cell_sleep_level_duration # list of dicts with keys 'time', 'cell', 'sleep_level'

    def loop(self):
        while True:
            if self.sim.env.now < self.sleep_delay_time:
                yield self.sim.wait(self.interval)
            if self.sim.env.now >= self.sleep_delay_time:
                for i in self.bedtime_stories:
                    if self.sim.env.now >= i['time']:
                        self.sim.cells[i['cell']].set_sleep_level(i['sleep_level'])
                        if self.sim.env.now >= i['time'] + i['duration'] or i['duration'] == -1:    # -1 means sleep forever
                            self.bedtime_stories.remove(i)           
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

    def get_cell_data_attached_UEs(self, cell):
        """
        Returns a list of data for each attached UE in a cell
        """
        data = []
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
        return data
    
    def get_cell_data_no_UEs(self, cell):
        """
        Returns a list of data for each cell with no attached UEs.
        """
        data = []
        UE = float('nan')
        serving_cell = float('nan')
        tm = self.sim.env.now                                           # current time
        sc_id = cell.i                                          # current UE serving_cell
        sc_sleep_mode = cell.get_sleep_mode()                   # current UE serving_cell sleep mode status
        sc_xy = cell.get_xyz()[:2]                              # current UE serving_cell xy position
        ue_id = float('nan')                                                    # current UE ID
        ue_xy = float('nan')                                        # current UE xy position
        d2sc = float('nan')                            # current UE distance to serving_cell
        tp = float('nan')             # current UE throughput ('fundamental')
        sc_power = cell.get_power_dBm()                         # current UE serving_cell transmit power
        sc_rsrp = float('nan')          # current UE rsrp from serving_cell
        neigh1_rsrp = float('nan')    # current UE neighbouring cell 1 rsrp
        neigh2_rsrp = float('nan')    # current UE neighbouring cell 2 rsrp
        noise = float('nan')                                      # current UE thermal noise
        sinr = float('nan')                                               # current UE sinr from serving_cell
        cqi = float('nan')                                                    # current UE cqi from serving_cell
        mcs = float('nan')                                  # current UE mcs for serving_cell

        # Get the above as a list
        data_list = [tm, sc_id, sc_sleep_mode, ue_id, d2sc, tp, sc_power, sc_rsrp, neigh1_rsrp, neigh2_rsrp, noise, sinr, cqi, mcs]

        # convert ndarrays to str or float
        for i, j in enumerate(data_list):
            if type(j) is np.ndarray:
                data_list[i] = float(j)

        # Write above to `data` list
        data.append(data_list)
        return data

    def get_data(self):
        # Create an empty list to store generated data
        all_data = []
        # Keep a list of column names to track
        columns = ["time", "serving_cell_id", "serving_cell_sleep_mode", "ue_id",
            "distance_to_cell(m)", "throughput(Mb/s)", "sc_power(dBm)","sc_rsrp(dBm)", "neighbour1_rsrp(dBm)", "neighbour2_rsrp(dBm)", "noise_power(dBm)", "sinr(dB)", "cqi", "mcs"]
        for cell in self.sim.cells:
            if len(cell.attached) != 0:
                # Get data for cells with attached UEs
                all_data.append(self.get_cell_data_attached_UEs(cell))
            if len(cell.attached) == 0:
                # Get data for cells with no attached UEs
                all_data.append(self.get_cell_data_no_UEs(cell))
        # Flatten the list
        all_data = [item for sublist in all_data for item in sublist]
        # Return the column names and data
        return columns, all_data
    


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

        # Find empty values in the dataframe and replace with NaN
        df1 = df1.replace(r'^\s*$', np.nan, regex=True)

        # Print df to screen
        print(df1)

        # (DEBUGGING TOOL) 
        # Print a view of the type of value in each position
        # --------------------------------------------------
        # df_value_type = df.applymap(lambda x: type(x).__name__)
        # print(df_value_type)

        # Write the MyLogger dataframe to TSV file
        df1.to_csv(logfile, sep="\t", index=False)

# END MyLogger class


def generate_ppp_points(sim, expected_pts=100, sim_radius=500.0, cell_centre_points=None, exclusion_radius=None):
    """
    Generates points distributed according to a homogeneous Poisson point process (PPP) in a disk, while ensuring that points are not placed within a certain radius of given coordinates.

    Parameters
    ----------
    sim : object
        An instance of the simulation.
    expected_pts : int, optional
        The expected number of points in the PPP. Default is 100.
    sim_radius : float, optional
        The radius of the simulation disk. Default is 500.0.
    cell_centre_points : np.ndarray, optional
        An Nx2 numpy array of the Cartesian coordinates of the exclusion zone centres. Default is None.
    exclusion_radius : float, optional
        A float value that specifies the radius in meters for which the function ensures that points are not placed within this radius of the coordinates in cell_centre_points. Default is None.

    Returns
    -------
    np.ndarray
        An Mx2 numpy array of the Cartesian coordinates of the PPP points.

    Notes
    -----
    The intensity (ie mean density) of the PPP is calculated as the expected number of points 
    divided by the area of the simulation disk. The simulation disk is centered at the origin.
    The generated points are distributed uniformly in the disk using polar coordinates and 
    then converted to Cartesian coordinates. The center of the disk is shifted to (0, 0).
    If cell_centre_points and exclusion_radius are provided, points within the exclusion zone are removed from the PPP.

    """
    sim_rng = sim.rng

    sim_radius = sim_radius
    xx0 = 0
    yy0 = 0
    areaTotal = np.pi * sim_radius ** 2

    lambda0 = expected_pts / areaTotal

    points = np.empty((0, 2))

    loop_count = 0
    remove_count = 0
    while points.shape[0] < expected_pts:
        loop_count += 1

        numbPoints = sim_rng.poisson(lambda0 * areaTotal)
        theta = 2 * np.pi * sim_rng.uniform(0, 1, numbPoints)
        rho = sim_radius * np.sqrt(sim_rng.uniform(0, 1, numbPoints))

        xx = rho * np.cos(theta)
        yy = rho * np.sin(theta)

        xx = xx + xx0
        yy = yy + yy0
        
        if cell_centre_points is not None and exclusion_radius is not None:
            dists = np.linalg.norm(cell_centre_points - np.array([xx, yy]), axis=1)
            indices = np.where(dists > exclusion_radius)[0]
            remove_count += (numbPoints - len(indices))
            xx = xx[indices]
            yy = yy[indices]
        
        points = np.vstack((points, np.column_stack((xx, yy))))

    points = points[:expected_pts]
    
    logging.info(f"The while loop ran {loop_count} times.")
    logging.info(f"{remove_count} points were removed from the exclusion zone.")
    
    return points


def hex_grid_setup(origin: tuple = (0, 0), isd: float = 500.0, sim_radius: float = 1000.0):
    """
    Create a hexagonal grid and plot it with a dashed circle.

    Parameters
    ----------
    origin : tuple of float, optional
        The center of the simulation area, by default (0, 0)
    isd : float, optional
        The distance between two adjacent hexagons (in meters), by default 500.0
    sim_radius : float, optional
        The radius of the simulation area (in meters), by default 1000.0

    Returns
    -------
    hexgrid_xy : numpy.ndarray
        A 2D array containing the x and y coordinates of the hexagonal grid.
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    Notes
    -----
    The hexagonal grid is created using the `create_hex_grid` function, and the
    resulting grid is plotted with matplotlib. The grid is centered at the origin
    and is scaled so that its diameter is `3 * isd + 500`. A dashed circle is also
    plotted with radius `sim_radius`.

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
    date = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.text(
        0.01, 0.005, f"{prespace}{author} {date}",
        ha='left', va='bottom', fontsize=fontsize, color=color,
        rotation=rotation,
        transform=fig.transFigure, alpha=alpha)


def plot_ues(sim, ue_ids=None, show_annotation=True):
    """
    Plots the location of UE objects in the simulation.

    Parameters
    ----------
    sim : object
        An instance of the simulation.
    ue_ids : list of int, optional
        A list of UE IDs to plot. If None, all UEs are plotted. Default is None.
    show_annotation : bool, optional
        If True, displays an annotation with the UE ID. Default is True.

    Returns
    -------
    None

    Notes
    -----
    The function extracts the UE objects with the given IDs from the simulation and plots their
    locations as red dots. The function also adds an annotation for each UE showing its ID if
    `show_annotation` is True. The location of the annotation is slightly offset from the UE
    location. If the `ue_ids` argument is None, all UEs in the simulation are plotted.

    """
    # FIXME - Add in switch to plot ues in a different color if they are in a cell

    if ue_ids is None:
        ue_objs_list = sim.UEs
        ue_ids = list(range(len(ue_objs_list)))
    else:
        ue_objs_list = [sim.UEs[i] for i in ue_ids]
    ue_x_list = [ue.xyz[0] for ue in ue_objs_list]
    ue_y_list = [ue.xyz[1] for ue in ue_objs_list]
    ue_xy_list = [ue.xyz[:2] for ue in ue_objs_list]
    plt.scatter(x=ue_x_list, y=ue_y_list, color='red', s=2.0)
    if show_annotation:
        for i in range(len(ue_ids)):
            plt.annotate(text=str(ue_ids[i]), xy=ue_xy_list[i], xytext=(3,-2), textcoords='offset points',
            fontsize=8, color='red', bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),)



def main(config_file):
    with open(config_file) as f:
            config = json.load(f)

    seed = config["seed"]
    isd = config["isd"]
    sim_radius = config["sim_radius"]
    power_dBm = config["power_dBm"]
    nues = config["nues"]
    until = config["until"]
    base_interval = config["base_interval"]
    new_power_dBm = config["new_power_dBm"]
    h_BS = config["h_BS"]
    h_UT = config["h_UT"]
    ue_noise_power_dBm = config["ue_noise_power_dBm"]
    scenario_delay = config["scenario_delay"]
    mme_cqi_limit = config["mme_cqi_limit"]
    mme_strategy = config["mme_strategy"]
    mme_anti_pingpong = config["mme_anti_pingpong"]
    mme_verbosity = config["mme_verbosity"]
    plot_ues = config.get("plot_ues")
    plot_ues_labels = config.get("plot_ues_labels")
    plot_ues_start = config["plot_ues_start"]
    plot_ues_end = config["plot_ues_end"]
    plot_author = config.get("plot_author")
    mcs_table_number = config["mcs_table_number"]
    

    # Create a simulator object
    sim = Simv2(rng_seed=seed)

    # Create instance of UMa-NLOS pathloss model
    pl_uma_nlos = UMa_pathloss(LOS=False)

    # Create the 19-cell hex-grid and place Cell instance at the centre
    sim_hexgrid_centres, hexgrid_plot = hex_grid_setup(isd=isd, sim_radius=sim_radius)
    for centre in sim_hexgrid_centres[:]:
        x, y = centre
        z = h_BS
        # Create the cell
        sim.make_cellv2(interval=base_interval*0.5,xyz=[x, y, z], power_dBm=power_dBm)

    # Generate UE positions using PPP
    ue_ppp = generate_ppp_points(sim=sim, expected_pts=nues, sim_radius=sim_radius)
    for i in ue_ppp:
        x, y = i
        ue_xyz = x, y, h_UT
        sim.make_UE(xyz=ue_xyz, reporting_interval=base_interval, pathloss_model=pl_uma_nlos).attach_to_strongest_cell_simple_pathloss_model()

    # Change the noise_power_dBm
    for ue in sim.UEs:
        ue.noise_power_dBm=ue_noise_power_dBm

    # Add the logger to the simulator
    custom_logger = MyLogger(sim, logging_interval=base_interval)
    sim.add_logger(custom_logger)

    # Add scenario to simulation
    change_outer_ring_power = ChangeCellPower(sim, delay=scenario_delay, new_power=new_power_dBm, interval=base_interval)
    sim.add_scenario(scenario=change_outer_ring_power)

    # Add MME for handovers
    default_mme = AMFv1(sim, cqi_limit=mme_cqi_limit, interval=base_interval,strategy=mme_strategy, anti_pingpong=mme_anti_pingpong,verbosity=mme_verbosity)
    sim.add_MME(mme=default_mme)

    # Plot UEs if desired (uncomment to activate)
    if plot_ues is True:
        if plot_ues_start is None:
            plot_ues_start = 0
        if plot_ues_end is None:
            plot_ues_end = len(sim.UEs)

        sim_ue_ids = list(range(plot_ues_start, plot_ues_end, 1))
        plot_ues(sim=sim, ue_ids=sim_ue_ids, show_annotation=plot_ues_labels)
        fig_timestamp(fig=hexgrid_plot, author=plot_author, timestamp=timestamp)
        fig_outfile_path = Path(logfile).with_suffix('.png')
        plt.savefig(fig_outfile_path)

    # Try setting sleep mode
    sim.cells[2].set_sleep_mode(mode=1)    # testing SLEEP_MODE

    # Run simulator
    sim.run(until=until)


if __name__ == '__main__':  # run the main script

    # Logic to allow the script to run inside VS Code while debugging
    if 'debugpy' in sys.modules:
        sys.argv = [sys.argv[0], '-c', '/Users/apw804/dev-02/EnergyModels/KISS/KISS_06a_1_config.json']


    # Create cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, required=True)

    # Create the args namespace
    args = parser.parse_args()

    # Set variables for later use
    script_name = Path(__file__).stem
    timestamp = datetime.now()
    timestamp_iso = timestamp.isoformat(timespec='seconds')
    logfile = '_'.join([script_name, 'logfile', timestamp_iso])+'.tsv'

    # Write input arguments to file for reference (uncomment to activate)
    # outfile = '_'.join([script_name,'config',timestamp_iso])
    # write_args_to_json(outfile=outfile, parse_args_obj=args)

    # Run the __main__
    main(config_file=args.config_file)