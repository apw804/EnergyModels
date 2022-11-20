# Kishan Sthankiya
# Basic energy model class based on test_f_callback_00.py
# Uses callback functions to compute the energy.
# Added dataclasses for small cells and macro cells.
# Example: python3 AIMM_simulator_QMUL_energy_00.py -ncells=5 -nues=20


import argparse
from dataclasses import dataclass

import numpy as np
from AIMM_simulator import Cell, UE, Scenario, Sim, from_dB


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

    def __init__(self, sim):
        """ Initialize variables which will accumulate energy totals.
        """

        self.sim = sim  # reference to the entire simulation!
        self.params_small_cell = SmallCellParameters()
        self.params_macro_cell = MacroCellParameters()
        self.cell_power_static = None  # baseline energy use
        self.cell_a_kW = 1.0  # slope
        self.ue_a_kW = 1.0e-3  # slope
        self.cell_power_totals = np.zeros(sim.get_ncells())
        self.ue_power_totals = np.zeros(sim.get_nues())

    def cell_power(self, cell):
        """
          Increment cell power consumption for one simulation timestep.
          Based on EARTH framework (10.1109/MWC.2011.6056691).
        """
        if cell.get_power_dBm() < 30:
            trx = self.params_small_cell
            self.cell_power_static = self.params_small_cell.power_static_watts
        else:
            trx = self.params_macro_cell
            self.cell_power_static = self.params_macro_cell.power_static_watts

        if cell.pattern is None:
            cell_antennas = 3  # Assume 3*120 degree antennas
        else:
            cell_antennas = 1  # If an array or function, assume it is unidirectional (for now)
        cell_sectors = 3  # Assuming 3 sectors. FIX when complex antennas implemented.

        n_trx = cell.n_subbands * cell_antennas * cell_sectors  # Number of transceiver chains
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

        self.cell_power_totals[cell.i] += cell.interval * (self.cell_power_static + n_trx * trx_power(p=trx_power_now))

    def ue_power(self, ue):
        """
          Increment UE energy usage for one simulation timestep.
        """
        self.ue_power_totals[ue.i] += ue.reporting_interval * self.ue_a_kW

    def f_callback(self, x, **kwargs):
        # print(kwargs)
        if isinstance(x, Cell):
            # print(f't={self.sim.env.now:.1f}: cell[{x.i}] (check from kwargs: {kwargs["cell_i"]}) energy={self.cell_energy_totals[x.i]:.0f}kW')
            self.cell_power(x)
        elif isinstance(x, UE):
            self.ue_power(x)


def test_01(ncells=4, nues=10, until=100.0):
    sim = Sim()
    for i in range(ncells):
        sim.make_cell(verbosity=0)
    for i in range(nues):
        ue = sim.make_UE(verbosity=1)
        ue.attach_to_nearest_cell()
    em = Energy(sim)
    for cell in sim.cells:
        cell.set_f_callback(em.f_callback, cell_i=cell.i)
    print(f'sim.get_nues()={sim.get_nues()}')
    for ue in sim.UEs:
        ue.set_f_callback(em.f_callback, ue_i=ue.i)
    scenario = Scenario(sim, verbosity=0)
    sim.add_scenario(scenario)
    sim.run(until=until)
    print(f'cell_energy_totals={em.cell_power_totals}watts')
    print(f'UE_energy_totals  ={em.ue_power_totals}kW')


if __name__ == '__main__':  # a simple self-test
    np.set_printoptions(precision=4, linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('-ncells', type=int, default=4, help='number of cells')
    parser.add_argument('-nues', type=int, default=10, help='number of UEs')
    parser.add_argument('-until', type=float, default=1000.0, help='simulation time')
    args = parser.parse_args()
    test_01(ncells=args.ncells, nues=args.nues, until=args.until)
