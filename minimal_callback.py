# Keith Briggs 2022-10-11
# Try out new callback functions for computing energy.
# 2022-11-02 16:37:08 Updated imports to work with package path logic (KS)
# Example: python3 test_f_callback_00.py -ncells=5 -nues=20

# 2022-11-05
# Kishan Sthankiya
# Trying a minimal callback function to understand how it works





import argparse

import numpy as np
from AIMM_simulator.core import Sim, Scenario, Cell, UE

def sim_callback(self, cell, **kwargs):
  self.cell = cell
  self.args = kwargs
  class_name = cell.__class__.__name__
  obj_name = cell.i
  a = 'This is a callback function.'
  b = 'It is attached to {} {}.'.format(class_name, obj_name)
  c = 'It accepts these arguments: {}.'.format(kwargs)

  return print(a,b,c, sep="\n")

def test_01(ncells=3,nues=3,until=1.0):
  sim=Sim()
  for i in range(ncells):
    sim.make_cell(verbosity=0)
  for i in range(nues):
    ue=sim.make_UE(verbosity=1)
    ue.attach_to_nearest_cell()
  for cell in sim.cells:
    cell.set_f_callback(sim_callback, cell=cell, fake_arg1=1, fake_arg2=2, fake_arg3=3)
  print(f'sim.get_nues()={sim.get_nues()}')
  scenario=Scenario(sim,verbosity=0)
  sim.add_scenario(scenario)
  sim.run(until=until)

if __name__=='__main__': # a simple self-test
  np.set_printoptions(precision=4,linewidth=200)
  parser=argparse.ArgumentParser()
  parser.add_argument('-ncells',type=int,default=3,help='number of cells')
  parser.add_argument('-nues',  type=int,default=3,help='number of UEs')
  parser.add_argument('-until', type=float,default=1.0,help='simulation time')
  args=parser.parse_args()
  test_01(ncells=args.ncells,nues=args.nues,until=args.until)

