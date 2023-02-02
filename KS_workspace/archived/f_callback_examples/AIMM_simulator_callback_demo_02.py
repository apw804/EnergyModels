# Keith Briggs 2022-11-19
# python3 AIMM_simulator_callback_demo.py
from functools import partial
from AIMM_simulator import Sim
class C:
  def f(self,cell,x,y):
    print(f'C.f called with self={self} cell={cell} x={x} y={y}')
sim=Sim()
cell=sim.make_cell()
f=partial(C.f,cell) # this passes "self" to C.f
cell.set_f_callback(f_callback=f,x='hello',y=1.0)
sim.run(until=100)