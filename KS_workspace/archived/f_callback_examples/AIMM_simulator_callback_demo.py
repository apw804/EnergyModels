# Keith Briggs 2022-11-19
# python3 AIMM_simulator_callback_demo.py
from AIMM_simulator import Sim,Cell
def f(cell,x):
  print(f'f called with cell={cell} x={x}')
sim=Sim()
cell=sim.make_cell()
cell.set_f_callback(f_callback=f,x='hello')
sim.run(until=100)