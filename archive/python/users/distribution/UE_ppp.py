# 2022-11-05
# Kishan Sthankiya
# Module to test Poisson Point Process for UE distribution in a geographical area.

import numpy as np
import matplotlib.pyplot as plt
from AIMM_simulator.core import Sim, Cell, Scenario, Logger, np_array_to_str


# Poisson Point Process: UE position as xy
def ppp(sim, nues, x_max, y_max, x_min=0, y_min=0):
    n_points = sim.rng.poisson(nues)  
    x = (x_max*np.random.uniform(0, 1,n_points)+x_min) # FIX: not a tuple
    y = (y_max*np.random.uniform(0, 1,n_points)+y_min) # FIX: not a tuple
    return np.stack((x, y), axis=1)


class MyScenario(Scenario):
  def loop(self,interval=0.1):
    while True:
      yield self.sim.wait(interval)

class MyLogger(Logger):
  def loop(self):
    self.f.write('#time\tcell\tUE\tx\ty\tthroughput\n')
    while True:
      for cell in self.sim.cells:
        for ue_i in cell.reports['cqi']:
          xy=self.sim.get_UE_position(ue_i)[:2]
          tp=np_array_to_str(cell.get_UE_throughput(ue_i))
          self.f.write(f't={self.sim.env.now:.1f}\tcell={cell.i}\tUE={ue_i}\tx={xy[0]:.0f}\ty={xy[1]:.0f}\ttp={tp}Mb/s\n')
      yield self.sim.wait(self.logging_interval)

# Plot the points for all UEs (for illustration only)
def plot_PPP(ppp_arr, ax_x_max, ax_y_max):
    plt.scatter(x=ppp_arr[:,0], y=ppp_arr[:,1], edgecolor='b', facecolor='none', alpha=0.5)
    plt.xlim(0, ax_x_max)
    plt.ylim(0, ax_y_max)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot()
    plt.show()

def test(n_cells, n_ues, plane_x_max=1000, plane_y_max=1000, until=500):
  sim=Sim(rng_seed=108)
  ue_ppp = ppp(sim=sim, nues=n_ues, x_max=plane_x_max, y_max=plane_y_max)
  plot_PPP(ppp_arr=ue_ppp, ax_x_max=plane_x_max, ax_y_max=plane_y_max)
  for i in range(n_cells): 
    sim.make_cell()
  for i, xy in enumerate(ue_ppp):
        ue_xyz = np.append(xy, 2.0)
        sim.make_UE(xyz=ue_xyz).attach_to_nearest_cell()
  sim.add_logger(MyLogger(sim,logging_interval=1))
  sim.add_scenario(MyScenario(sim))
  sim.run(until=until)


test(n_cells=4, n_ues=50)