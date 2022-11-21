# Kishan Sthankiya
# Basic energy model class based on test_f_callback_00.py
# Uses callback functions to compute the energy.
# Example: python3 AIMM_simulator_QMUL_energy_00.py -ncells=5 -nues=20


import argparse

import numpy as np
from AIMM_simulator import Cell, UE, Scenario, Sim


class Energy:
    """
    Defines a complete self-contained system energy model.
    """

    def __init__(self, sim):
        """ Initialize variables which will accumulate energy totals.
        """
        self.sim = sim  # reference to the entire simulation!
        self.cell_energy_static = 10.0  # baseline energy use
        self.cell_a_kW = 1.0  # slope
        self.ue_a_kW = 1.0e-3  # slope
        self.cell_energy_totals = np.zeros(sim.get_ncells())
        self.ue_energy_totals = np.zeros(sim.get_nues())

    def cell_energy(self, cell):
        """
          Increment cell energy usage for one simulation timestep.
          The implementation here which adds a term proportional
          to the number of attached UEs is for illustration only.
        """
        n_attached = cell.get_nattached()
        self.cell_energy_totals[cell.i] += cell.interval * (self.cell_energy_static + self.cell_a_kW * n_attached)

    def ue_energy(self, ue):
        """
          Increment UE energy usage for one simulation timestep.
        """
        self.ue_energy_totals[ue.i] += ue.reporting_interval * self.ue_a_kW

    def f_callback(self, x, **kwargs):
        # print(kwargs)
        if isinstance(x, Cell):
            # print(f't={self.sim.env.now:.1f}: cell[{x.i}] (check from kwargs: {kwargs["cell_i"]}) energy={self.cell_energy_totals[x.i]:.0f}kW')
            self.cell_energy(x)
        elif isinstance(x, UE):
            self.ue_energy(x)


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
    print(f'cell_energy_totals={em.cell_energy_totals}kW')
    print(f'UE_energy_totals  ={em.ue_energy_totals}kW')


if __name__ == '__main__':  # a simple self-test
    np.set_printoptions(precision=4, linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('-ncells', type=int, default=4, help='number of cells')
    parser.add_argument('-nues', type=int, default=10, help='number of UEs')
    parser.add_argument('-until', type=float, default=1000.0, help='simulation time')
    args = parser.parse_args()
    test_01(ncells=args.ncells, nues=args.nues, until=args.until)
