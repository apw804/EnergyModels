# KISS (Keep It Simple Stupid!) v3
# Now gets the dataframe output as floats instead of np.ndarrays
# FIXME - need to finish integrating the NR_5G_standard... max cell throughput
# Scenario: Runs Richard's suggestion to reduce the cell power of the outer ring of cells.
# FIXME - add the AMF to limit the UEs that are attached to Cells based on whether they have a CQI>0
# Added the updated MCS table2 from PHY.py-data-procedures to override NR_5G_standard_functions.MCS_to_Qm_table_64QAM
# Adding sleep mode by subclassing Cell and Sim
# Refactored components from ReduceCellPower_14.py: CellEnergyModel
# THIS IS KISS_PROFILED.PY


import argparse
import dataclasses
import json
import os
import datetime
from types import NoneType
import _bisect

import numpy as np
import pandas as pd
from AIMM_simulator import *
from hexalattice.hexalattice import *
import utils_kiss


# Fix custom imports
bisect_left = _bisect.bisect_left
get_timestamp = utils_kiss.get_timestamp

# Get a timestamp for the current time and date
date_now = get_timestamp(date_only=True)
time_now = get_timestamp(time_only=True)


def create_logfile_path(config_dict, debug_logger: bool = False):
    """Create a path for the log file based on the config parameters"""
    project_root_dir = config_dict['project_root_dir']
    script_name = config_dict['script_name']
    experiment_description = config_dict['experiment_description']
    if debug_logger:
        logfile_name = "_".join(
            [
                time_now,
                "debug",
                script_name
            ])
    else:
        # if experiment_description starts with "test_", then remove the "test_" from the acronym, but prefix the acronym with "test_"
        if experiment_description.startswith("test_"):
            acronym = "test_" + ''.join(config_dict['experiment_description'])
        else:
            acronym = ''.join(word[0] for word in config_dict['experiment_description'].split('_'))
        

        logfile_name = "_".join(
            [
                time_now,
                acronym,
                # Add the experiment_version to the logfile name
                "v" + str(config_dict['experiment_version']).replace(".", "_"),
                "s" + str(config_dict['seed']),
                "p" + str(config_dict['variable_cell_power_dBm']) + "dBm",
                "n_var_cells" + str(config_dict['n_variable_power_cells']),
            ])
    
    # if config_dict["experiment_description"] begins with test_, then add `_test` to the logfile_path
    if config_dict["experiment_description"].startswith("test_"):
        logfile_path = f"{project_root_dir}/_test/data/output/{experiment_description}/{date_now}/".replace(".", "_")
    else:
        logfile_path = f"{project_root_dir}/data/output/{experiment_description}/{date_now}/".replace(".", "_")
    
    if not os.path.exists(logfile_path):
        os.makedirs(os.path.dirname(logfile_path), exist_ok=True)

    logfile_path = os.path.join(logfile_path, logfile_name)
        
    return logfile_path



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
            NR_5G_standard_functions.MCS_to_Qm_table_64QAM = kiss_phy_data_procedures.mcs_table_1     # same as LTE
        elif mcs_table_number == 2:
            NR_5G_standard_functions.MCS_to_Qm_table_64QAM = kiss_phy_data_procedures.mcs_table_2     # 5G NR; 256QAM
        elif mcs_table_number == 3:
            NR_5G_standard_functions.MCS_to_Qm_table_64QAM = kiss_phy_data_procedures.mcs_table_3     # 5G NR; 64QAM LowSE/RedCap (e.g. IoT devices)
        elif mcs_table_number == 4:
            NR_5G_standard_functions.MCS_to_Qm_table_64QAM = kiss_phy_data_procedures.mcs_table_4     # 5G NR; 1024QAM
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
            self.sim.mme.attach_ues_to_n_best_rsrp(orphaned_ues)

    
    def get_sleep_mode(self):
        """
        Return the sleep_mode for the Cellv2.
        """
        return self.sleep_mode
    

    def get_cell_throughput(self):
        """
        Returns the throughput of a cell in the current timestep.
        """
        cell_throughput = 0.0
        for ue_i in self.attached:
            ue_tp_check = self.get_UE_throughput(ue_i)
            if ue_tp_check is not None:
                cell_throughput += ue_tp_check
        return cell_throughput
    
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
    
    def make_cellv2(self, **cell_kwargs):
            ''' 
            Convenience function: make a new Cellv2 instance and add it to the simulation; parameters as for the Cell class. Return the new Cellv2 instance.).
            '''
            self.cells.append(Cellv2(self,**cell_kwargs))
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


    def attach_ues_to_n_best_rsrp(self, ues, n=0):
        """
        Accepts a list of UEs. Attaches UE to the cell that gives it the Nth best rsrp cell, where N is the rank of the RSRP value. 
        
        N = 0 is the best RSRP value.
        N = 1 is the second best RSRP value.
        If N is greater than the number of cells in the simulation, the UE will be attached to the cell with the best RSRP value.

        Prints to stdout, the UEs that will be handed over and the cell that the UE will be handed over to.
        """
        # Print to stdout, the UEs that will be handed over
        print(f'Handing over UEs: {ues}')
        for ue_i in list(ues):
            ue = self.sim.UEs[ue_i]
            # If N is greater than the number of cells in the simulation, the UE will be attached to the cell with the best RSRP value.
            if n > len(self.sim.cells):
                n = 0
                # Get the cell that the UE will be handed over to.
                celli=self.sim.get_best_rsrp_cell(ue_i)

            # If N is less than or equal to the number of cells in the simulation, the UE will be attached to the cell with the Nth best RSRP value.
            if n <= len(self.sim.cells):
                # Get the cell that the UE will be handed over to.
                celli, cell_rsrp=self.sim.loggers[0].get_neighbour_cell_rsrp_rank(ue_id=ue_i, neighbour_rank=n)

                # Print to stdout, the cell that the UE will be handed over to.
                print(f'Handing over UE[{ue_i}] to Cell[{celli}]')
                # Detach the UE from it's current serving cell.
                old_cell = ue.serving_cell.i
                ue.detach()
                print(f'UE[{ue_i}] has been detached from Cell[{old_cell}].')
                # Attach the UE to the new cell.
                ue.attach(self.sim.cells[celli])
                print(f'UE[{ue_i}] has been attached to Cell[{celli}].')
                # Send RSRP and CQI reports to the new cell.
                ue.send_rsrp_reports() # make sure we have reports immediately
                ue.send_subband_cqi_report()
                print(f'UE[{ue_i}] has sent RSRP and CQI reports to Cell[{celli}].')

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

    def __init__(self, cell: Cellv2, interval=1.0):
        """
        Initialize variables.
        """
        # Check the cell power_dBm when initialising. Set as macro or small cell accordingly.
        self.cell = cell

        # Log the cell id to make sure that only the owner Cell instance can update via a callback function
        self.cell_id = self.cell.i

        kiss_debugger.debug("Attaching CellEnergyModel to Cell[%s]", self.cell.i)
        if self.cell.get_power_dBm() >= 30.0:
            kiss_debugger.debug("Cell[%s] transmit power > 30 dBm.", self.cell.i)

            self.cell_type = 'MACRO'
            kiss_debugger.debug("Cell[%s] type set to %s.", self.cell.i, self.cell_type)

            self.params = MacroCellParameters()
            kiss_debugger.debug("Cell[%s] params set to %s.",
                          cell.i, self.params.__class__.__name__)

        else:
            kiss_debugger.debug("Cell[%s] transmit power < 30 dBm.", self.cell.i)

            self.cell_type = 'SMALL'
            kiss_debugger.debug("Cell[%s] type set to %s.", self.cell.i, self.cell_type)

            self.params = SmallCellParameters()
            kiss_debugger.debug("Cell[%s] params set to %s.",
                          self.cell.i, self.params.__class__.__name__)

        # List of params to store
        self.CELL_POWER_OUT_DBM_MAX = self.params.p_max_dbm
        kiss_debugger.debug("Cell[%s] P_out_Cell_max_dBm: %s.",
                      self.cell.i, self.CELL_POWER_OUT_DBM_MAX)

        self.SECTOR_TRX_CHAIN_POWER_KILOWATTS_STATIC = self.params.power_static_watts / \
            1000  # baseline energy use
        kiss_debugger.debug("Cell[%s] P_out_Sector_TRXchain_static_kW: %s.", self.cell.i,
                      self.SECTOR_TRX_CHAIN_POWER_KILOWATTS_STATIC)

        # The load based power consumption
        self.SECTOR_TRX_CHAIN_POWER_KILOWATTS_DYNAMIC = self.trx_chain_power_dynamic_kW()
        kiss_debugger.debug("Cell[%s] P_out_Sector_TRXchain_dynamic_kW: %s.", self.cell.i,
                      self.SECTOR_TRX_CHAIN_POWER_KILOWATTS_DYNAMIC)

        # Calculate the starting cell power
        self.cell_power_kW = self.params.sectors * self.params.antennas * (
            self.SECTOR_TRX_CHAIN_POWER_KILOWATTS_STATIC + self.SECTOR_TRX_CHAIN_POWER_KILOWATTS_DYNAMIC)
        kiss_debugger.debug(
            "Starting power consumption for Cell[%s] (kW): %s", self.cell.i, self.cell_power_kW)

        # END of INIT

    def from_dBm_to_watts(self, x):
        """Converts dBm (decibel-milliwatt) input value to watts"""
        return from_dB(x) / 1000

    def get_power_out_per_trx_chain_watts(self, cell_power):
        """
        Takes an input value for a cell power output in dBm.
        Returns the power output of a single TRX chain in Watts.
        A TRX chain consists of an antenna, power amplifier, rf unit and baseband unit.
        """
        return self.from_dBm_to_watts(cell_power)

    def trx_chain_power_dynamic_kW(self):
        """
        Returns the power consumption (in kW), per sector / antenna.
        """
        cell_p_out_dBm = self.cell.get_power_dBm()

        if cell_p_out_dBm > self.CELL_POWER_OUT_DBM_MAX:
            raise ValueError('Power cannot exceed the maximum cell power!')

        # Get current TRX chain output power in watts
        trx_p_out_watts = self.get_power_out_per_trx_chain_watts(cell_p_out_dBm)

        # Sanity check that other input values are in decimal form
        p_rf_watts = self.params.power_rf_watts
        p_bb_watts = self.params.power_baseband_watts

        # Calculate the Power Amplifier power consumption in watts
        p_pa_watts = trx_p_out_watts / \
            (self.params.eta_pa * from_dB(1 - self.params.loss_feed_db))

        # Calculate the value of `P_ue_plus_C_watts` given the number of UEs multiplex by the base station
        if self.cell.get_nattached() == 0:
            p_ue_plus_C_watts = 0.0
        else:
            p_ue_plus_C_watts = trx_p_out_watts / self.cell.get_nattached()

        # p_ue_watts

        # Calculate power consumptions of a single TRX chain (watts)
        p_consumption_watts = p_pa_watts + p_rf_watts + p_bb_watts

        # Calculate losses (ratio)
        p_losses_ratio = (1 - self.params.loss_dc) * \
            (1 - self.params.loss_mains) * (1 - self.params.loss_cool)

        # Get the power output per TRX chain (watts)
        p_out_TRX_chain_watts = p_consumption_watts / p_losses_ratio

        # Power output per TRX chain (kW)
        p_out_TRX_chain_kW = p_out_TRX_chain_watts / 1000

        # Update the instance stored value
        self.SECTOR_TRX_CHAIN_POWER_KILOWATTS_DYNAMIC = p_out_TRX_chain_kW

        return p_out_TRX_chain_kW
    
    def get_cell_sleep_mode(self):
        """
        Returns the cell sleep mode.
        """
        return self.cell.sleep_mode

    def reset_energy_model_params(self):
        """
        Resets the energy model parameters to default values based on the __init__ power_dBm set.
        """
        if self.cell_type == 'MACRO':
            self.params = MacroCellParameters()
        elif self.cell_type == 'SMALL':
            self.params = SmallCellParameters()
    

    def update_cell_power_kW(self):
        """
        Updates the cell power consumption (in kW).
        """
        # Reset the energy model parameters to defaults
        self.reset_energy_model_params()

        # Get the cell sleep mode
        cell_sleep_mode = self.get_cell_sleep_mode()

        # If the cell sleep_mode is between 1 and 4, then the cell is in sleep mode
        if cell_sleep_mode in range(1, 5):

            # Set a placeholder for sleep_time
            sleep_time = 0

            # Get the cell interval in microseconds
            time_interval = self.cell.interval * 1e6

            # Get the OFDM symbol time
            ofdm_symbol_time = calculate_ofdm_symbol_time(self.cell.bw_MHz, mu=1)

            # Calculate the maximum number of whole OFDM symbols that can be transmitted in the time interval
            n_ofdm_symbols_overall = int(time_interval / ofdm_symbol_time)

            # Determine the sleep mode and set the sleep time parameters
            mode = cell_sleep_mode
            if mode == 1:
                sleep_time = random_time_interval(self.cell.sim, min_time_us=ofdm_symbol_time, max_time_us=1000)
            elif mode == 2:
                sleep_time = random_time_interval(self.cell.sim, min_time_us=5000, max_time_us=10000)
            elif mode == 3:
                sleep_time = random_time_interval(self.cell.sim, min_time_us=50000, max_time_us=100000)
            elif mode == 4:
                sleep_time = random_time_interval(self.cell.sim, min_time_us=500000, max_time_us=1000000)

            # Calculate the active time ratio in microseconds
            ratio_active = (time_interval - sleep_time) / time_interval

            # Calculate the sleep ratio
            sleep_ratio = sleep_time / time_interval

            # Store the ratio of whole OFDM symbols that can be transmitted in the active time as a cell attribute
            self.cell.sleep_mode_ratio_active = ratio_active        
            # FIXME - this is a hack to get the ratio active to the cell. 
            # Need to find a better way to do this and scale throughput accordingly (for slep sleep modes >1)

            # The amount of static power consumed by the TRX chain is the same for all sleep modes
            static_power = self.SECTOR_TRX_CHAIN_POWER_KILOWATTS_STATIC

            # The ratio of active time is the amount of time that the cell is using FULL power, so there is NO SCALING for this amount of time
            active_power = self.trx_chain_power_dynamic_kW() * ratio_active

            # Sleep modes turn hardware componenets off so here we `zero` these by scaling to the ratio of active time
            # Determine the sleep mode and set the power consumption parameters accordingly
            power_rf_watts_scaled = self.params.power_rf_watts * ratio_active
            power_baseband_watts_scaled = self.params.power_baseband_watts * ratio_active
            eta_pa_scaled = self.params.eta_pa * ratio_active
            loss_dc_scaled = self.params.loss_dc * ratio_active

            if mode == 1:
                self.params = dataclasses.replace(self.params, power_rf_watts=power_rf_watts_scaled)
            elif mode == 2:
                self.params = dataclasses.replace(self.params, power_rf_watts=power_rf_watts_scaled, power_baseband_watts=power_baseband_watts_scaled)
            elif mode == 3:
                self.params = dataclasses.replace(self.params, power_rf_watts=power_rf_watts_scaled, power_baseband_watts=power_baseband_watts_scaled, eta_pa=eta_pa_scaled)
            elif mode == 4:
                self.params = dataclasses.replace(self.params, power_rf_watts=power_rf_watts_scaled, power_baseband_watts=power_baseband_watts_scaled, eta_pa=eta_pa_scaled, loss_dc=loss_dc_scaled)
            
            # The ratio of SLEEP time is the amount of time that the cell is using SCALED power, so there is LESS POWER CONSUMPTION for this amount of time
            sleep_power = self.trx_chain_power_dynamic_kW() * sleep_ratio

            # Calculate the total power consumption of the cell
            self.cell_power_kW = self.params.sectors * self.params.antennas * (static_power + active_power + sleep_power)
            
            kiss_debugger.debug(
                'Cell[%s] power consumption has been updated to: %s', self.cell.i, self.cell_power_kW)
            return

        # If the cell sleep_mode is 0, then the cell is in active mode
        # Update the cell power as normal
        self.cell_power_kW = self.params.sectors * self.params.antennas * (
            self.SECTOR_TRX_CHAIN_POWER_KILOWATTS_STATIC + self.trx_chain_power_dynamic_kW())
        kiss_debugger.debug(
            'Cell[%s] power consumption has been updated to: %s', self.cell.i, self.cell_power_kW)
        

    def get_cell_power_kW(self, time):
        """
        Returns the power consumption (in kW) of the cell at a given time.
        """
        if time == 0:
            kiss_debugger.debug(
                'Cell[%s] power consumption at t=0 is %s', self.cell.i, self.cell_power_kW)
            return self.cell_power_kW
        else:
            self.update_cell_power_kW()
            kiss_debugger.debug(
                'Cell[%s] power consumption at t=%s is %s', self.cell.i, time, self.cell_power_kW)
            return self.cell_power_kW

    def f_callback(self, x, **kwargs):
        if isinstance(x, Cellv2):
            if x.i == self.cell_id:
                self.update_cell_power_kW()
            else:
                kiss_debugger.warning(
                    'Cell[%s] is trying to update the Cell[%s] energy model.', x.i, self.cell_id)
                raise ValueError(
                    'Cells can only update their own energy model instances! Check the cell_id.')
            
# End class Energy

class ChangeCellPower(Scenario):
    """
    Changes the power_dBm of the specified list of cells (default random cell ) after a specified delay time (if provided), relative to t=0.
    """

    def __init__(self, sim, interval=0.5, cells=None, n_cells=None, delay=None, new_power=None):
        """
        Initializes an instance of the ChangeCellPower class.

        Parameters:
        -----------
        sim : SimPy.Environment
            The simulation environment object.
        interval : float, optional
            The time interval between each power change. Default is 0.5.
        cells : list of int, optional
            The list of cell indices to change power. Default is random_cell (based on the seed in the simulation)
        n_cells : int, optional
            The number of random cells to change power. Default is None.
        delay : float, optional
            The delay time before changing the cell powers. Default is None.
        new_power : float, optional
            The new power_dBm to set for the specified cells. Default is None.
        """
        self.target_cells = None
        self.delay_time = delay
        self.new_power = new_power
        self.sim = sim
        self.interval = interval
        self.random_cell = self.sim.rng.integers(low=0, high=len(self.sim.cells))
        self.outer_ring = [0, 1, 2, 3, 6, 7, 11, 12, 15, 16, 17, 18]
        if self.target_cells is None:
            if n_cells is not None and n_cells > 0:
                self.target_cells = self.sim.rng.choice(len(self.sim.cells), n_cells, replace=False)
            elif cells is not None:
                    self.target_cells = cells
            else:
                self.target_cells = self.random_cell

    def loop(self):
        """
        The main loop of the scenario, which changes the power of specified cells after a delay.
        """
        while True:
            while self.sim.env.now >= self.delay_time:
                if isinstance(self.target_cells, list) or isinstance(self.target_cells, np.ndarray):
                    for i in self.target_cells:
                        self.sim.cells[i].set_power_dBm(self.new_power)
                elif isinstance(self.target_cells, int):
                    self.sim.cells[self.target_cells].set_power_dBm(self.new_power)
                else:
                    raise ValueError(
                        'The target_cells parameter must be a list, array, or integer.')
                yield self.sim.env.timeout(self.interval)





class RemoveRandomCell(Scenario):
    """
    Randomly selected cell from the simulation environment and remove after delay time (if provided), relative to t=0.
    """

    def __init__(self, sim, delay=None, interval=0.1):
        """
        Initializes an instance of the RemoveRandomCell class.

        Parameters:
        -----------
        sim : SimPy.Environment
            The simulation environment object.
        interval : float, optional
            The time interval between each cell removal. Default is 0.1.
        delay : float, optional
            The delay time before removing the cell. Default is None.
        """
        self.sim = sim
        self.interval = interval
        self.delay_time = delay

    def loop(self):
        """
        The main loop of the scenario, which removes a randomly selected cell after a delay.
        """
        # Select a random cell index using the sim.rng object
        cell_index = self.sim.rng.integers(low=0, high=len(self.sim.cells))
        # Print to stdout, the selection of the cell for removal
        print(f'Cell[{cell_index}] has been selected for removal.')
        while True:
            if self.sim.env.now < self.delay_time:
                yield self.sim.wait(self.interval)
            if self.sim.env.now > self.delay_time:
                if self.sim.cells[cell_index] in self.sim.cells:
                    # Get a list of all users in the cell
                    ues = self.sim.cells[cell_index].attached
                    # For each UE, handover to the best rsrp cell
                    # This will also update the ue.serving_cell attribute and print messages to stdout
                    self.sim.mme.attach_ues_to_n_best_rsrp(ues)      
                    # Remove the cell from the simulation environment
                    self.sim.cells.pop(cell_index)
                    # Print to stdout, the removal of the cell
                    print(f'Cell[{cell_index}] has been removed from the simulation.')
                    # Print the handovers, reporting the new cell ID for all ues 
                    for ue in ues:
                        print(f'UE[{ue}] has been handed over from Cell{cell_index} to Cell[{self.sim.UEs[ue].serving_cell.i}].')
                yield self.sim.wait(self.interval)
            yield self.sim.wait(self.interval)


class SetCellSleep(Scenario):
    """
    A scenario that sets the sleep level of specified cells at specified times.
    """
    def __init__(self, sim, interval=1.0, time_cell_sleep_level_duration=None, delay=0):
        self.sleep_delay_time = delay
        self.sim = sim
        self.interval = interval
        if time_cell_sleep_level_duration is None:
            print('WARNING: No cells specified for SetCellSleep. No cells will be put to sleep.')
            self.bedtime_stories = []
        else:
            self.bedtime_stories = []
            for item in time_cell_sleep_level_duration:
                if not isinstance(item, dict):
                    raise TypeError(f'Expected dict for time_cell_sleep_level_duration, got {type(item)}')
                if {'time', 'cell', 'sleep_level', 'sleep_duration'} != item.keys():
                    raise KeyError(f'Expected keys "time", "cell", "sleep_level", and "sleep_duration" in time_cell_sleep_level_duration, got {item.keys()}')
                if not isinstance(item['time'], (int, float)):
                    raise TypeError(f'Expected int or float for time, got {type(item["time"])}')
                if not isinstance(item['cell'], int):
                    raise TypeError(f'Expected int for cell, got {type(item["cell"])}')
                if item['cell'] not in [x for x, j in enumerate(self.sim.cells)]:
                    raise KeyError(f'Cell {item["cell"]} not found in simulation environment.')
                if not isinstance(item['sleep_level'], int):
                    raise TypeError(f'Expected int for sleep_level, got {type(item["sleep_level"])}')
                if not 0 <= item['sleep_level'] <= 4:
                    raise ValueError(f'Expected sleep_level to be between 0 and 4, got {item["sleep_level"]}')
                if not isinstance(item['sleep_duration'], (int, float)):
                    raise TypeError(f'Expected int or float for sleep_duration, got {type(item["sleep_duration"])}')
                self.bedtime_stories.append(item)

    def loop(self):
        while True:
            if self.sim.env.now < self.sleep_delay_time:
                yield self.sim.wait(self.interval)
            for item in self.bedtime_stories:
                if self.sim.env.now >= item['time']:
                    self.sim.cells[item['cell']].set_sleep_mode(item['sleep_level'])
                    if item['sleep_duration'] == -1:
                        yield self.sim.wait(self.interval)
                    elif item['sleep_duration'] - self.sim.env.now <= 0: 
                        self.bedtime_stories.remove(item)
                yield self.sim.wait(self.interval)


class MyLogger(Logger):

    def __init__(self, *args,cell_energy_models=None, logfile_path=None, **kwargs):
        self.dataframe = None
        self.cell_energy_models = cell_energy_models
        self.logfile_path = logfile_path
        super().__init__(*args, **kwargs)

    def get_cqi_to_mcs(self, cqi):
        """
        Returns the MCS value for a given CQI value. Copied from `NR_5G_standard_functions.CQI_to_64QAM_efficiency`
        """
        if type(cqi) is NoneType:
            return np.nan
        else:
            return max(0,min(28,int(28*cqi/15.0)))



    def get_neighbour_cell_rsrp_rank(self, ue_id):
        """
        Returns a list of tuples containing the ID and RSRP of neighbouring cells for the given UE,
        with the highest RSRP value at position 0 for each cell in self.sim.cells.

        Parameters
        ----------
        ue_id : int
            The ID of the UE.

        Returns
        -------
        list
            A list of tuples containing the ID and RSRP of neighbouring cells for the given UE.
            
        Notes
        -----
        This function returns the neighbouring cell RSRPs, where index 0 is not the original serving cell.
        """
        # Create a place to hold the results that will contain tuples of (cell_id, rsrp) and can be efficiently sorted by rsrp
        neighbour_cell_rsrp = []
 
        # Loop through all cells
        for cell in self.sim.cells:
            cell_id = cell.i
            # Skip the serving cell
            if cell_id == self.sim.UEs[ue_id].serving_cell.i:
                continue
            # Skip cells that have no entry for the given UE
            if ue_id not in cell.reports["rsrp"]:
                continue
            # Skip cells that have no RSRP value for the given UE
            ue_rsrp = cell.reports["rsrp"][ue_id][1]
            if ue_rsrp == -np.inf:
                continue
            # Add the cell to the list if it doesn't exist already
            if not any(cell_id == cell_tuple[1] for cell_tuple in neighbour_cell_rsrp):
                # Insert (rsrp, cell_id) into the neighbour_cell_rsrp list, such that the list is sorted by rsrp in descending order
                neighbour_cell_rsrp.insert(bisect_left(neighbour_cell_rsrp, (ue_rsrp, cell_id)), (ue_rsrp, cell_id))

        return neighbour_cell_rsrp



    def get_cell_data_attached_UEs(self, cell):
        """
        Returns a list of data for each attached UE in a cell
        """
        data = []
        for attached_ue_id in cell.attached:
            UE = self.sim.UEs[attached_ue_id]
            serving_cell = UE.serving_cell
            if serving_cell.power_dBm == -np.inf or serving_cell.power_dBm <= 0.0:
                serving_cell.reports["rsrp"].clear()                        # clear the serving cell["rsrp"] reports if the serving cell is not transmitting
                                                                            # FIXME - using a dict comprehension doesn't clear the dictionary properly for some reason (e.g. {key: 0 for key in serving_cell.reports["rsrp"]}  )
                serving_cell.reports["throughput_Mbps"].clear()             # clear the serving cell["throughput"] reports if the serving cell is not transmitting
                UE.sinr_dB = 0.0                                          # clear the UE["sinr_dB"] reports if the serving cell is not transmitting
            cell_energy_model = self.cell_energy_models[serving_cell.i]
            seed = self.sim.seed
            neigh_rsrp_array = self.get_neighbour_cell_rsrp_rank(UE.i)
            tm = self.sim.env.now                                           # current time
            sc_id = serving_cell.i                                          # current UE serving_cell
            sc_sleep_mode = serving_cell.get_sleep_mode()                   # current UE serving_cell sleep mode status
            sc_xy = serving_cell.get_xyz()[:2]                              # current UE serving_cell xy position
            ue_id = UE.i                                                    # current UE ID
            ue_xy = UE.get_xyz()[:2]                                        # current UE xy position
            d2sc = np.linalg.norm(sc_xy - ue_xy)                            # current UE distance to serving_cell
            ue_tp = serving_cell.get_UE_throughput(attached_ue_id)          # current UE throughput ('fundamental')
            sc_power = serving_cell.get_power_dBm()                         # current UE serving_cell transmit power
            sc_rsrp = serving_cell.get_rsrp(ue_id)                          # current UE rsrp from serving_cell
            neigh1_rsrp = neigh_rsrp_array[-1][0]    
            neigh2_rsrp = neigh_rsrp_array[-2][0]    
            noise = UE.noise_power_dBm                                      # current UE thermal noise
            sinr = UE.sinr_dB                                               # current UE sinr from serving_cell
            cqi = UE.cqi                                                    # current UE cqi from serving_cell
            mcs = self.get_cqi_to_mcs(cqi)                                  # current UE mcs for serving_cell
            cell_tp = serving_cell.get_cell_throughput()                    # current UE serving_cell throughput
            cell_power_kW = cell_energy_model.get_cell_power_kW(tm)         # current UE serving_cell power consumption
            cell_ee = (cell_tp * 1e6) / (cell_power_kW * 1e3)               # current UE serving_cell energy efficiency
            cell_se = (cell_tp * 1e6) / (serving_cell.bw_MHz * 1e6)         # current UE serving_cell spectral efficiency

            # Get the above as a list
            data_list = [seed, tm, sc_id, sc_sleep_mode, ue_id, d2sc, ue_tp, sc_power, sc_rsrp, neigh1_rsrp, neigh2_rsrp, noise, sinr, cqi, mcs, cell_tp, cell_power_kW, cell_ee, cell_se]

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
        seed= self.sim.seed
        UE = float('nan')
        serving_cell = float('nan')
        cell_energy_model = self.cell_energy_models[cell.i]
        tm = self.sim.env.now                                       # current time
        sc_id = cell.i                                              # current cell
        sc_sleep_mode = cell.get_sleep_mode()                       # current cell sleep mode status
        sc_xy = cell.get_xyz()[:2]                                  # current cell xy position
        ue_id = float('nan')                                        # UE ID
        ue_xy = float('nan')                                        # UE xy position
        d2sc = float('nan')                                         # distance to serving_cell
        ue_tp = float('nan')                                        # UE throughput ('fundamental')
        sc_power = cell.get_power_dBm()                             # current UE serving_cell transmit power
        sc_rsrp = float('nan')                                      # current UE rsrp from serving_cell
        neigh1_rsrp = float('nan')                                  # current UE neighbouring cell 1 rsrp
        neigh2_rsrp = float('nan')                                  # current UE neighbouring cell 2 rsrp
        noise = float('nan')                                        # current UE thermal noise
        sinr = float('nan')                                         # current UE sinr from serving_cell
        cqi = float('nan')                                          # current UE cqi from serving_cell
        mcs = float('nan')                                          # current UE mcs for serving_cell
        cell_tp = cell.get_cell_throughput()                        # current UE serving_cell throughput
        cell_power_kW = cell_energy_model.get_cell_power_kW(tm)     # current UE serving_cell power consumption
        cell_ee = ((cell_tp * 1e6) / (cell_power_kW * 1e3)) * 1e6   # current UE serving_cell energy efficiency
        cell_se = (cell_tp * 1e6) / (cell.bw_MHz * 1e6)     # current UE serving_cell spectral efficiency

        # Get the above as a list
        data_list = [seed, tm, sc_id, sc_sleep_mode, ue_id, d2sc, ue_tp, sc_power, sc_rsrp, 
                     neigh1_rsrp, neigh2_rsrp, noise, sinr, cqi, mcs, cell_tp, cell_power_kW, 
                     cell_ee, cell_se]

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
        columns = ["seed", "time", "serving_cell_id", "serving_cell_sleep_mode", "ue_id",
            "distance_to_cell(m)", "ue_throughput(Mb/s)", "sc_power(dBm)","sc_rsrp(dBm)", 
            "neighbour1_rsrp(dBm)", "neighbour2_rsrp(dBm)", "noise_power(dBm)", "sinr(dB)", 
            "cqi", "mcs", "cell_throughput(Mb/s)", "cell_power(kW)", "cell_ee(bits/J)", 
            "cell_se(bits/Hz)"]
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

        # print(f'Finalize time={self.sim.env.now}')

        # Run routine for final time step
        self.run_routine(ignore_index=True)

        # Create a copy of the final DataFrame
        df = self.dataframe.copy()

        # Sort the data by time, UE_id then cell_id
        df.sort_values(['time','ue_id', 'serving_cell_id'], ascending=[True, True, True])
        df1 = df.copy()

        # Convert all NaN and -np.inf values to 0.0
        df1 = df1.replace([np.inf, -np.inf], np.nan)
        df1 = df1.fillna(0.0)

        # Find empty values in the dataframe and replace with NaN
        df1 = df1.replace(r'^\s*$', np.nan, regex=True)

        # Print df to screen
        # print(df1)

        # (DEBUGGING TOOL) 
        # Print a view of the type of value in each position
        # --------------------------------------------------
        # df_value_type = df.applymap(lambda x: type(x).__name__)
        # print(df_value_type)

        # Write the MyLogger dataframe to TSV file
        df1.to_csv(self.logfile_path, sep="\t", index=False, mode='w')




# END MyLogger class


def calculate_ofdm_symbol_time(channel_bandwidth, mu):
    """
    Calculate the time for one OFDM symbol in 5G NR from the channel bandwidth, and subcarrier spacing.

    Parameters:
    channel_bandwidth (float): channel bandwidth in MHz
    mu (int): subcarrier spacing exponent (0, 1, 2, 3)

    Returns:
    float: the time for one OFDM symbol in microseconds
    """
    subcarrier_spacing = get_subcarrier_spacing(mu)
    num_subcarriers = channel_bandwidth * 1e6 // subcarrier_spacing
    symbol_duration = 1 / (num_subcarriers * subcarrier_spacing)
    return symbol_duration * 1e6


def get_subcarrier_spacing(mu):
    """
    Get the subcarrier spacing in Hz for a given mu.

    Parameters:
    mu (int): subcarrier spacing exponent (0, 1, 2, 3)

    Returns:
    float: the subcarrier spacing in Hz
    """
    if mu == 0:
        return 15e3
    elif mu == 1:
        return 30e3
    elif mu == 2:
        return 60e3
    elif mu == 3:
        return 120e3
    else:
        raise ValueError("Invalid mu value. Must be 0, 1, 2, or 3.")

def calculate_symbols_per_second(ofdm_symbol_duration):
    """
    Calculate the number of OFDM symbols in 1 second given the OFDM symbol duration.

    Parameters:
    ofdm_symbol_duration (float): the duration of one OFDM symbol in seconds

    Returns:
    float: the number of OFDM symbols per second
    """
    return 1 / ofdm_symbol_duration

def random_time_interval(sim, min_time_us, max_time_us):
    """
    Returns a random time interval between min_time_us and max_time_us (inclusive)
    based on the provided rng_seed value.

    Args:
        rng_seed (int): Seed value for the random number generator.
        min_time_us (float): Minimum time interval in microseconds.
        max_time_us (float): Maximum time interval in microseconds.

    Returns:
        float: Random time interval in seconds.
    """
    rng = sim.rng
    time_interval_us = rng.uniform(min_time_us, max_time_us)  # Uniform distribution between min and max
    return time_interval_us 



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
    If cell_centre_points and exclusion_radius are provided, points within the exclusion zone 
    are removed from the PPP.

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
    
    kiss_debugger.debug(f"The while loop ran {loop_count} times.")
    kiss_debugger.debug(f"{remove_count} points were removed from the exclusion zone.")
    
    return points


def hex_grid_setup(origin: tuple = (0, 0), isd: float = 500.0, sim_radius: float = 1000.0, plot: bool = False, watermark: bool = False):
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
    plot : bool, optional
        Whether to plot the hexagonal grid and the dashed circle, by default False
    watermark : bool, optional
        Whether to add watermark numbers to each hexagon, by default False

    Returns
    -------
    hexgrid_xy : numpy.ndarray
        A 2D array containing the x and y coordinates of the hexagonal grid.
    fig : matplotlib.figure.Figure or None
        The matplotlib Figure object containing the plot, or None if plot=False.

    Notes
    -----
    The hexagonal grid is created using the `create_hex_grid` function, and the
    resulting grid is plotted with matplotlib. The grid is centered at the origin
    and is scaled so that its diameter is `3 * isd + 500`. A dashed circle is also
    plotted with radius `sim_radius`.

    """
    if plot:
        fig, ax = plt.subplots()
    else:
        fig = None
        ax = None
    
    from hexalattice.hexalattice import create_hex_grid

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

    if plot:
        circle_dashed = plt.Circle(
            origin, sim_radius, fill=False, linestyle='--', color='r')

        ax.add_patch(circle_dashed)
        ax.scatter(hexgrid_x, hexgrid_y, marker='2')
        
        if watermark:
            # Add watermark numbers to each hexagon
            for i, (x, y) in enumerate(hexgrid_xy):
                ax.text(x + isd / 3 - 0.3e3, y - isd /3 + 0.33e3, str(i), fontsize=10, ha='center', va='center', color='purple', alpha=0.63, weight='bold')
        
        # Factor to set the x,y-axis limits relative to the isd value.
        ax_scaling = 3 * isd + 500
        ax.set_xlim([-ax_scaling, ax_scaling])
        ax.set_ylim([-ax_scaling, ax_scaling])
        ax.set_aspect('equal')
    
        return hexgrid_xy, fig
    else:
        if watermark:
            # Add watermark numbers to each hexagon
            for i, (x, y) in enumerate(hexgrid_xy):
                plt.text(x + isd / 3 - 0.3e3, y - isd /3 + 0.33e3, str(i), fontsize=10, ha='center', va='center', color='purple', alpha=0.63, weight='bold')
        
        return



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
    date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.text(
        0.01, 0.005, f"{prespace}{author} {date}",
        ha='left', va='bottom', fontsize=fontsize, color=color,
        rotation=rotation,
        transform=fig.transFigure, alpha=alpha)


def plot_ues_fig(sim, ue_ids_start=None, ue_ids_end=None, show_labels=True, labels_start=None, labels_end=None):

    # FIXME - Add in switch to plot ues in a different color if they are in a cell
    if ue_ids_start is None and ue_ids_end is None:
            ue_ids_start = 0
            ue_ids_end = len(sim.UEs)
    if ue_ids_start is not None and ue_ids_end is None:
        ue_ids_end = ue_ids_start
    if ue_ids_start is None and ue_ids_end is not None:
        ue_ids_start = ue_ids_end
    
    if ue_ids_start == ue_ids_end:
        ue_ids = ue_ids_start
    elif ue_ids_start < ue_ids_end:
        ue_ids = list(range(ue_ids_start, ue_ids_end))
    elif ue_ids_start > ue_ids_end:
        # Throw an error
        raise ValueError("ue_ids_start must be less than ue_ids_end")
    
    if isinstance(ue_ids, int):
        ue_objs_list = [sim.UEs[ue_ids]]
    if isinstance(ue_ids, list) or isinstance(ue_ids, np.ndarray):
        ue_objs_list = [sim.UEs[i] for i in ue_ids]
    else:
        ue_objs_list = [sim.UEs[i] for i in ue_ids]
    ue_x_list = [ue.xyz[0] for ue in ue_objs_list]
    ue_y_list = [ue.xyz[1] for ue in ue_objs_list]
    ue_xy_list = [ue.xyz[:2] for ue in ue_objs_list]
    plt.scatter(x=ue_x_list, y=ue_y_list, color='red', alpha= 0.3, s=2.0)
    if show_labels:
        if labels_start is None and labels_end is None:
            labels_start = 0
            labels_end = len(ue_ids)
        if labels_end is None:
            labels_end = labels_start + 1 # Only label the first labels_start ue_ids
        for i in range(labels_start, labels_end):
            plt.annotate(text=str(ue_ids[i]), xy=ue_xy_list[i], xytext=(3,-2), textcoords='offset points',
            fontsize=8, color='red', bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),)


def main(config_dict):
    seed = config_dict["seed"]
    sim_show_params = config_dict["sim_show_params"]
    isd = config_dict["isd"]
    sim_radius = config_dict["sim_radius"]
    power_dBm = config_dict["constant_cell_power_dBm"]
    variable_cell_power_dBm = config_dict["variable_cell_power_dBm"]
    n_variable_power_cells = config_dict["n_variable_power_cells"]
    variable_power_target_cells_list = config_dict["variable_power_target_cells_list"]
    nues = config_dict["nues"]
    until = config_dict["until"]
    base_interval = config_dict["base_interval"]
    h_BS = config_dict["h_BS"]
    h_UT = config_dict["h_UT"]
    ue_noise_power_dBm = config_dict["ue_noise_power_dBm"]
    scenario_delay = config_dict["scenario_delay"]
    SetCellSleep_bedtime_stories = config_dict["scenario_SetCellSleep_params"]["SetCellSleep_bedtime_stories"]
    mme_cqi_limit = config_dict["mme_cqi_limit"]
    mme_strategy = config_dict["mme_strategy"]
    mme_anti_pingpong = config_dict["mme_anti_pingpong"]
    mme_verbosity = config_dict["mme_verbosity"]
    plotting = config_dict["plotting"]
    plot_cell_id_watermark = config_dict["plot_cell_id_watermark"]
    plot_ues = config_dict["plot_ues"]
    plot_ues_start = config_dict["plot_ues_start"]
    plot_ues_end = config_dict["plot_ues_end"]
    plot_ues_show_labels = config_dict["plot_ues_show_labels"]
    plot_ues_labels_start = config_dict["plot_ues_show_labels_start"]
    plot_ues_labels_end = config_dict["plot_ues_show_labels_end"]
    plot_author = config_dict.get("plot_author")
    mcs_table_number = config_dict["mcs_table_number"]

    # Create a log file path
    data_output_logfile_path = create_logfile_path(config_dict)

    # Create a simulator object
    sim = Simv2(rng_seed=seed, show_params=sim_show_params)
    sim.seed = seed

    # Create instance of UMa-NLOS pathloss model
    pl_uma_nlos = UMa_pathloss(LOS=False)

    # Create the 19-cell hex-grid and place Cell instance at the centre
    sim_hexgrid_centres, hexgrid_plot = hex_grid_setup(isd=isd, sim_radius=sim_radius, plot=plotting, watermark=plot_cell_id_watermark)
    for centre in sim_hexgrid_centres[:]:
        x, y = centre
        z = h_BS
        # Create the cell
        sim.make_cellv2(interval=base_interval*0.1,xyz=[x, y, z], power_dBm=power_dBm)  # interval=base_interval*0.1 (scaled down to see if rsrp dictionaries get cleared)

    # Create a dictionary of cell-specific energy models
    cell_energy_models_dict = {}
    for cell in sim.cells:
        cell_energy_models_dict[cell.i] = (CellEnergyModel(cell))
        cell.set_f_callback(cell_energy_models_dict[cell.i].f_callback(cell))

    # Generate UE positions using PPP
    ue_ppp = generate_ppp_points(sim=sim, 
                                 expected_pts=nues, 
                                 sim_radius=sim_radius,)
    for i in ue_ppp:
        x, y = i
        ue_xyz = x, y, h_UT
        sim.make_UE(xyz=ue_xyz, reporting_interval=base_interval, pathloss_model=pl_uma_nlos, verbosity=0).attach_to_strongest_cell_simple_pathloss_model()

    # Change the noise_power_dBm
    for ue in sim.UEs:
        ue.noise_power_dBm=ue_noise_power_dBm

    # Add the logger to the simulator
    custom_logger = MyLogger(sim,
                             logging_interval = base_interval, 
                             cell_energy_models = cell_energy_models_dict, 
                             logfile_path = ".".join([data_output_logfile_path, "tsv"]))
    sim.add_logger(custom_logger)



    # Add scenarios to simulation
    reduce_random_cell_power = ChangeCellPower(
        sim, 
        n_cells=n_variable_power_cells,
        delay=scenario_delay,
        new_power=variable_cell_power_dBm, 
        interval=base_interval
        )
    
    reduce_centre_cell_power = ChangeCellPower(
        sim,
        delay=scenario_delay,
        cells=variable_power_target_cells_list[0],
        new_power=variable_cell_power_dBm, 
        interval=base_interval
        )
    
    change_outer_ring_power = ChangeCellPower(
        sim, 
        delay=scenario_delay, 
        new_power=variable_cell_power_dBm, 
        interval=base_interval
        )
    
    remove_random_cell = RemoveRandomCell(
        sim, 
        delay=scenario_delay, 
        interval=base_interval
        )
    
    set_cell_sleep = SetCellSleep(
        sim, 
        interval=base_interval, 
        time_cell_sleep_level_duration=SetCellSleep_bedtime_stories
        )

    # Activate scenarios
    sim.add_scenario(scenario=reduce_centre_cell_power)

    # Add MME for handovers
    default_mme = AMFv1(sim, cqi_limit=mme_cqi_limit, interval=base_interval,strategy=mme_strategy, anti_pingpong=mme_anti_pingpong,verbosity=mme_verbosity)
    sim.add_MME(mme=default_mme)

    # Plot UEs if desired (uncomment to activate)
    if plot_ues:
        plot_ues_fig(sim=sim, ue_ids_start=plot_ues_start, ue_ids_end=plot_ues_end ,show_labels=plot_ues_show_labels, labels_start=plot_ues_labels_start, labels_end=plot_ues_labels_end)
        fig_timestamp(fig=hexgrid_plot, author=plot_author)
        file_name = os.path.splitext(data_output_logfile_path)[0]
        fig_outfile_path = file_name + '.png'
        plt.savefig(fig_outfile_path)


    # Run simulator
    sim.run(until=until)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run kiss.py against a specified config value.')

    parser.add_argument(
        '-c',
        '--config-file',
        type=str,
        required=True,
        default='KISS/_test/data/input/kiss/test_kiss.json'
        )
    
    args = parser.parse_args()

    # Load the config file
    config_file = args.config_file
    with open(config_file, "r") as f:
        config = json.load(f)

    # If debug_logging is enabled, create a logger object
    if config["debug_logging"]:
        try:
            from logging_kiss import get_logger
            # If successful, create a logger object
            if callable(get_logger):
                # Create a log file path
                kiss_debugger_outfile_path = create_logfile_path(
                    config_dict=config, 
                    debug_logger=True
                    )
                # Create a logger object
                kiss_debugger = get_logger(
                    logger_name="kiss_debugger",
                    logfile_path=kiss_debugger_outfile_path,
                    log_level=config["debug_logging_level"],
                    )
        except ImportError:
            print("logging_kiss not installed. debug_logging disabled.")
            config["debug_logging"] = False
    else:
        # If debug_logging is disabled, create a dummy logger object
        import logging
        class DummyLogger(logging.Logger):
            def __init__(self, name):
                super().__init__(name, level=logging.NOTSET)

            def _log(self, *args, **kwargs):
                pass

        # Create a DummyLogger object
        kiss_debugger = DummyLogger("kiss_debugger")


    # Import plt if plotting is enabled
    if config["plotting"]:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed. Plotting disabled.")
            config["plotting"] = False

    # Import kiss_phy_data_procedures if mcs_table_number is set
    if config["mcs_table_number"] is not None:
        try:
            import kiss_phy_data_procedures
        except ImportError:
            print("kiss_phy_data_procedures not installed. mcs_table_number disabled.")
            config["mcs_table_number"] = None
    
    # If line_profiler is enabled, import line_profiler
    if config["line_profiler"]:
        try:
            import line_profiler
            if callable(line_profiler):
                # Method to profile a function
                def profile_method(method):
                    def wrapper(*args, **kwargs):
                        lp = line_profiler.LineProfiler()
                        lp_wrapper = lp(method)
                        lp_wrapper(*args, **kwargs)
                        lp.print_stats()
                    return wrapper
                # Use `@profile_method` to profile a function
        except ImportError:
            print("line_profiler not installed. line_profiler disabled.")
            config["line_profiler"] = False

    # Start timer
    start_time = time()

    # Run main
    main(config_dict=config)

    # End timer
    end_time = time()

    # Log total time taken
    kiss_debugger.info("Total time taken (s): {}".format(end_time - start_time))

