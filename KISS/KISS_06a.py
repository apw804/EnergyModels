# KISS (Keep It Simple Stupid!) v3
# Now gets the dataframe output as floats instead of np.ndarrays
# FIXME - need to finish integrating the NR_5G_standard... max cell throughput
# Scenario: Runs Richard's suggestion to reduce the cell power of the outer ring of cells.
# FIXME - add the AMF to limit the UEs that are attached to Cells based on whether they have a CQI>0
# Added the updated MCS table2 from PHY.py-data-procedures to override NR_5G_standard_functions.MCS_to_Qm_table_64QAM
# Adding sleep mode by subclassing Cell and Sim
# Refactored components from ReduceCellPower_14.py: CellEnergyModel
# With the JSON bits for config


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

class NRPHYFrame:
    def __init__(self, numerology, num_slots):
        """
        Initialize a new 5G NR PHY frame with the given numerology and number of slots.
        """
        self.numerology = numerology
        self.num_slots = num_slots
        self.slot_duration = 1 / (2 ** self.numerology)
        self.symbol_duration = self.slot_duration / 14
        self.num_symbols_per_slot = 14 * (2 ** self.numerology)
        self.num_symbols = self.num_symbols_per_slot * self.num_slots
        self.num_subcarriers = 12 * (2 ** self.numerology)
        self.num_sc_per_rb = 12
        self.num_rbs_per_slot = self.num_subcarriers // self.num_sc_per_rb
        self.num_rbs = self.num_rbs_per_slot * self.num_slots
        self.num_re_per_rb = 14
        self.num_re_per_slot = self.num_re_per_rb * self.num_rbs_per_slot
        self.num_re = self.num_re_per_slot * self.num_slots
    
    def get_slot_duration(self):
        """
        Get the duration of a single slot in seconds.
        """
        return self.slot_duration
    
    def get_symbol_duration(self):
        """
        Get the duration of a single symbol in seconds.
        """
        return self.symbol_duration
    
    def get_num_symbols_per_slot(self):
        """
        Get the number of symbols per slot.
        """
        return self.num_symbols_per_slot
    
    def get_num_symbols(self):
        """
        Get the total number of symbols in the frame.
        """
        return self.num_symbols
    
    def get_num_subcarriers(self):
        """
        Get the total number of subcarriers in the frame.
        """
        return self.num_subcarriers
    
    def get_num_rbs_per_slot(self):
        """
        Get the number of resource blocks per slot.
        """
        return self.num_rbs_per_slot
    
    def get_num_rbs(self):
        """
        Get the total number of resource blocks in the frame.
        """
        return self.num_rbs
    
    def get_num_re_per_rb(self):
        """
        Get the number of resource elements per resource block.
        """
        return self.num_re_per_rb
    
    def get_num_re_per_slot(self):
        """
        Get the total number of resource elements per slot.
        """
        return self.num_re_per_slot
    
    def get_num_re(self):
        """
        Get the total number of resource elements in the frame.
        """
        return self.num_re
    
    def get_re_frequency(self, re_idx):
        """
        Get the frequency (subcarrier index) of the given resource element index.
        """
        rb_idx = re_idx // self.num_re_per_rb
        sc_idx = (re_idx % self.num_re_per_rb) % self.num_sc_per_rb
        return rb_idx * self.num_sc_per_rb + sc_idx
    
    def get_re_time(self, re_idx):
        """
        Get the time (symbol index) of the given resource element index.
        """
        slot_idx = re_idx // self.num_re_per_slot
        sym_idx = (re_idx % self.num_re_per_slot) % self.num_symbols_per_slot
        return slot_idx * self.symbol_duration * self.num_symbols_per_slot + sym_idx * self.symbol_duration

# End of NRPHYFrame

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

    def __init__(self, *args, energy_model=None, sleep_mode=None, mcs_table_number=None, **kwargs):   # [How to use *args and **kwargs with 'peeling']
        # Set Cellv2 energy_model
        if not isinstance(energy_model, NoneType):
            self.energy_model = CellEnergyModel(self.i)
        if isinstance(sleep_mode, NoneType):
            self.sleep_mode = 0
        else:
            self.sleep_mode = sleep_mode
        if isinstance(mcs_table_number, NoneType):
            self.mcs_table_number = mcs_table_number
        super().__init__(*args, **kwargs)

    
    def get_mcs_table_number(self):
        return self.mcs_table_number

    def set_mcs_table(self, mcs_table_number):
        """
        Changes the lookup table used by NR_5G_standard_functions.MCS_to_Qm_table_64QAM
        """
        if isinstance(mcs_table_number, NoneType):
            mcs_table_number = self.get_mcs_table_number()
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
    def __init__(self, **kwargs):
        self.orphaned_ues = []
        super().__init__(**kwargs)
    
    def make_cellv2(self, **kwargs):
            ''' 
            Convenience function: make a new Cellv2 instance and add it to the simulation; parameters as for the Cell class. Return the new Cellv2 instance.).
            '''
            self.cells.append(Cellv2(self,**kwargs))
            xyz=self.cells[-1].get_xyz()
            self.cell_locations=np.vstack([self.cell_locations,xyz])
            return self.cells[-1]


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

    def __init__(self,
                 cell_id,
                 interval=1.0):
        """
        Initialize variables.
        """
        # Check the cell power_dBm when initialising. Set as macro or small cell accordingly.
        self.cell = self.sim.cells[cell_id]

        # Log the cell id to make sure that only the owner Cell instance can update via a callback function
        self.cell_id = cell_id
        logging.debug("Attaching CellEnergyModel to Cell[%s]", self.cell.i)
        if self.cell.get_power_dBm() >= 30.0:
            logging.debug("Cell[%s] transmit power > 30 dBm.", self.cell.i)

            self.cell_type = 'MACRO'
            logging.debug("Cell[%s] type set to %s.", cell_id, self.cell_type)

            self.params = MacroCellParameters()
            logging.debug("Cell[%s] params set to %s.",
                          cell.i, self.params.__class__.__name__)

        else:
            logging.debug("Cell[%s] transmit power < 30 dBm.", self.cell.i)

            self.cell_type = 'SMALL'
            logging.debug("Cell[%s] type set to %s.", self.cell.i, self.cell_type)

            self.params = SmallCellParameters()
            logging.debug("Cell[%s] params set to %s.",
                          self.cell.i, self.params.__class__.__name__)

        # List of params to store
        self.CELL_POWER_OUT_DBM_MAX = self.params.p_max_dbm
        logging.debug("Cell[%s] P_out_Cell_max_dBm: %s.",
                      self.cell.i, self.CELL_POWER_OUT_DBM_MAX)

        self.SECTOR_TRX_CHAIN_POWER_KILOWATTS_STATIC = self.params.power_static_watts / \
            1000  # baseline energy use
        logging.debug("Cell[%s] P_out_Sector_TRXchain_static_kW: %s.", self.cell.i,
                      self.SECTOR_TRX_CHAIN_POWER_KILOWATTS_STATIC)

        # The load based power consumption
        self.SECTOR_TRX_CHAIN_POWER_KILOWATTS_DYNAMIC = self.trx_chain_power_dynamic_kW()
        logging.debug("Cell[%s] P_out_Sector_TRXchain_dynamic_kW: %s.", self.cell.i,
                      self.SECTOR_TRX_CHAIN_POWER_KILOWATTS_DYNAMIC)

        # Calculate the starting cell power
        self.cell_power_kW = self.params.sectors * self.params.antennas * (
            self.SECTOR_TRX_CHAIN_POWER_KILOWATTS_STATIC + self.SECTOR_TRX_CHAIN_POWER_KILOWATTS_DYNAMIC)
        logging.debug(
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

    def update_cell_power_kW(self):
        # First update the cell power
        self.cell_power_kW = self.params.sectors * self.params.antennas * (
            self.SECTOR_TRX_CHAIN_POWER_KILOWATTS_STATIC + self.trx_chain_power_dynamic_kW())
        logging.debug(
            'Cell[%s] power consumption has been updated to: %s', self.cell.i, self.cell_power_kW)

    def get_cell_power_kW(self, time):
        if time == 0:
            logging.debug(
                'Cell[%s] power consumption at t=0 is %s', self.cell.i, self.cell_power_kW)
            return self.cell_power_kW
        else:
            self.update_cell_power_kW()
            logging.debug(
                'Cell[%s] power consumption at t=%s is %s', self.cell.i, time, self.cell_power_kW)
            return self.cell_power_kW

    def f_callback(self, x, **kwargs):
        if isinstance(x, Cell):
            if x.i == self.cell_id:
                self.update_cell_power_kW()
            else:
                logging.warning(
                    'Cell[%s] is trying to update the Cell[%s] energy model.', x.i, self.cell_id)
                raise ValueError(
                    'Cells can only update their own energy model instances! Check the cell_id.')
            
# End class Energy


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
    """
    Custom logger class to log the UE's RSRP and CQI values.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataframe = None

    def get_cqi_to_mcs(self, cqi):
        """
        Returns the MCS value for a given CQI value. Copied from `NR_5G_standard_functions.CQI_to_64QAM_efficiency`
        """
        if type(cqi) is NoneType:
            return np.nan
        else:
            return max(0,min(28,int(28*cqi/15.0)))


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
                yield self.sim.wait(self.logging_interval / 2)
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
        print(df1)

        # Write the dataframe to TSV file
        df1.to_csv(self.logfile, sep="\t", index=False)

# END MyLogger class


def generate_ppp_points(sim, expected_pts=100, sim_radius=500.0):
    """
    Generates points distributed according to a homogeneous Poisson point process (PPP) in a disk.

    Parameters
    ----------
    sim : object
        An instance of the simulation.
    expected_pts : int, optional
        The expected number of points in the PPP. Default is 100.
    sim_radius : float, optional
        The radius of the simulation disk. Default is 500.0.

    Returns
    -------
    np.ndarray
        An Nx2 numpy array of the Cartesian coordinates of the PPP points.

    Notes
    -----
    The intensity (ie mean density) of the PPP is calculated as the expected number of points 
    divided by the area of the simulation disk. The simulation disk is centered at the origin.
    The generated points are distributed uniformly in the disk using polar coordinates and 
    then converted to Cartesian coordinates. The center of the disk is shifted to (0, 0).

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


from datetime import datetime

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


def load_config(config_file_path):
    """
    Load a configuration from a JSON file.

    Parameters
    ----------
    config_file_path : str
        The path to the JSON file containing the configuration.

    Returns
    -------
    dict
        The configuration loaded from the JSON file.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.

    json.JSONDecodeError
        If the specified file path contains invalid JSON.

    Examples
    --------
    >>> config_file_path = "/path/to/config.json"
    >>> config = load_config(config_file_path)
    """
    with open(config_file_path, "r") as config_file:
        config = json.load(config_file)
    return config

def update_keys(config, base_val, target_word):
    """
    Recursively updates any keys in the config dictionary that contain the target word 
    in their name with the specified base value.
    Parameters
    ----------
    config : dict
        The dictionary to update.
    base_val : float
        The base value to use for the updated keys.
    target_word : str
        The target word to search for in the key names.

    Returns
    -------
    dict
        The updated dictionary.
    """
    for key, value in config.items():
        if isinstance(value, dict):
            update_keys(value, base_val, target_word)
        elif target_word in key:
            config[key] = base_val
    return config



def main(config_file_path):
    config = load_config(config_file_path)
    base_interval = config["experiment_settings"]["base_interval"]
    update_keys(config, base_val=base_interval, target_word="interval")
    for key in config["simv2_settings"]["params"].keys():
        update_keys(config, base_val=config['simv2_settings']['params'][key], target_word=key)

    # See what the nested dictionary looks like
    pp_config = json.dumps(config, indent=2)  
    print(pp_config)

    # Unpack the nested dictionary to kwargs
    experiment_kwargs = config["experiment_settings"]
    hexgrid_kwargs = config["hexgrid_settings"]
    simv2_kwargs = config["simv2_settings"]
    cellv2_kwargs = config["cellv2_settings"]
    UMa_pathloss_kwargs = config["UMa_pathloss_model_settings"]
    UE_kwargs = config["UE_settings"]
    MyLogger_kwargs = config["MyLogger_settings"]
    ChangeCellPower_kwargs = config["ChangeCellPower_settings"]
    AMFv1_kwargs = config["AMFv1_settings"]

    # Create Simv2 object
    if isinstance(experiment_kwargs["seeds"], list):
        print("Hasn't been figured out yet!")
        pass
    elif isinstance(experiment_kwargs["seeds"], int):
        simv2_kwargs["rng_seed"] = experiment_kwargs["seeds"]
        sim = Simv2(**simv2_kwargs)
        return sim
     
    # Create the UMa_pathloss object
    UMa_model = UMa_pathloss(**UMa_pathloss_kwargs)

    # Create 19-cell hexgrid
    sim_hexgrid_centres, hexgrid_plot = hex_grid_setup(**hexgrid_kwargs)
    for centre in sim_hexgrid_centres[:]:
        cellv2_kwargs["xyz"][:2] = centre.tolist()
        cellv2_kwargs["xyz"][2] = cellv2_kwargs["h_BS"]
        # Create the cell
        sim.make_cellv2(**cellv2_kwargs)


    for cell in sim.cells:
        # Quick and simple labelling of cell_ids
        cell_id = cell.i
        cell_x = cell.xyz[0]
        cell_y = cell.xyz[1]
        plt.annotate(cell_id, (cell_x, cell_y), color='blue', alpha=0.3)

    # Generate UE positions using PPP
    ue_ppp = generate_ppp_points(sim=sim, expected_pts=experiment_kwargs["expected_pts"], sim_radius=hexgrid_kwargs["sim_radius"])
    for i in ue_ppp:
        UE_kwargs["xyz"][:2] = i
        UE_kwargs["xyz"][2] = simv2_kwargs["params"]["h_UT"]
        sim.make_UE(**UE_kwargs).attach_to_strongest_cell_simple_pathloss_model()

    # Change the noise_power_dBm for all UEs to -118dBm
    for ue in sim.UEs:
        ue.noise_power_dBm=-118.0

    # Add the logger to the simulator
    custom_logger = MyLogger(sim, **MyLogger_kwargs)
    sim.add_logger(custom_logger)

    # Add scenario to simulation
    change_outer_ring_power = ChangeCellPower(sim=sim, **ChangeCellPower_kwargs)
    sim.add_scenario(scenario=change_outer_ring_power)

    # Add MME for handovers
    default_mme = AMFv1(sim=sim, **AMFv1_kwargs)
    sim.add_MME(mme=default_mme)

    # Plot UEs if desired (uncomment to activate)
    sim_ue_ids = [ue.i for ue in sim.UEs]
    plot_ues(sim=sim, show_annotation=False)
    fig_timestamp(fig=hexgrid_plot, author='Kishan Sthankiya')
    fig_outfile_path = Path(logfile).with_suffix('.png')
    plt.savefig(fig_outfile_path)

    # Try setting sleep mode
    sim.cells[2].set_sleep_mode(mode=1)    # testing SLEEP_MODE

    # Run the simulation
    sim.run(until=experiment_kwargs["until"])


if __name__ == '__main__':  # run the main script

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
    main('/Users/apw804/dev-02/EnergyModels/KISS_06_config_template.json')
