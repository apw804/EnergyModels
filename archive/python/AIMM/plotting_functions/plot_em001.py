# Kishan Sthankiya
# 2022-10-13 21:46:54 
# Plotter written for em001.py

import os

import matplotlib as plt
import pandas as pd
from AIMM_simulator_core_17 import Logger


class Em001Logger(Logger):
  '''
  Custom logger written for energy model 001 (em001.py).
  Written in the context of the cell.

  CustomParameters
  ----------------

  '''

  # Constructor for QmEnergyLogger with additional parameter
  def __init__(self, *args, **kwargs):
    # Calling the parent constructor using ClassName.__init__()
    Logger.__init__(s, *args, **kwargs)
    # Adding the cols and main_dataframe
    self.cols = (
      'time'
      'cell'
      'n_UEs'
      'a_tp(Mbps)'
      'EC(kW)'
      'EE'
    )
    self.main_dataframe = pd.DataFrame(data=None, columns=list(self.cols))

  def append_row(self, new_row):
    temp_df = pd.DataFrame(data=[new_row], columns=list(self.cols))
    self.main_dataframe = pd.concat([self.main_dataframe, temp_df])

  def loop(s):
    # Write to stdout
    s.f.write("#time\tcell\tn_UEs\ta_tp (Mb/s)\tEC (kW)\t\tEE (Mbps/kW)\n")
    while True:
      for cell in s.sim.cells:
        tm = s.sim.env.now
        cell_i = cell.i

  # Constructor for QmEnergyLogger with additional parameter
  def __init__(self, *args, **kwargs):
    # Calling the parent constructor using ClassName.__init__()
    Logger.__init__(s, *args, **kwargs)
    # Adding the cols and main_dataframe
    self.cols = (
      'time'
      'cell'
      'n_UEs'
      'a_tp(Mbps)'
      'EC(kW)'
      'EE'
    )
    self.main_dataframe = pd.DataFrame(data=None, columns=list(self.cols))

  def append_row(self, new_row):
    temp_df = pd.DataFrame(data=[new_row], columns=list(self.cols))
    self.main_dataframe = pd.concat([self.main_dataframe, temp_df])

  def loop(s):
    # Write to stdout
    s.f.write("#time\tcell\tn_UEs\ta_tp (Mb/s)\tEC (kW)\t\tEE (Mbps/kW)\n")
    while True:
      for cell in s.sim.cells:
        tm = s.sim.env.now
        cell_i = cell.i
        n_UEs = len(s.sim.cells[cell_i].attached)
        tp = 0.00
        for ue_i in s.sim.cells[cell_i].attached:
          tp += s.sim.cells[cell_i].get_UE_throughput(ue_i)
        ec = QM_Energy_001.cell_energy(self, cell)
        # Calculate the energy efficiency
        if tp == 0.0:
          ee = 0.0
        else:
          ee = tp / ec
        # Write to stdout
        s.f.write(
          f"t={tm:.1f}\t{cell_i}\t{n_UEs}\t{tp:.2f}\t\t{ec:.2f}\t\t{ee:.3f}\n"
        )
        # Write these variables to the main_dataframe
        row = (tm, cell_i, n_UEs, tp, ec, ee)
        new_row = [numpy.round(each_arr, decimals=3) for each_arr in row]
        s.append_row(new_row)
        yield s.sim.wait(s.logging_interval)

  def finalize(s):
      cwd = os.getcwd()
      path = cwd + os.path.basename(__file__)
      s.main_dataframe.to_csv(path)

      # Separate out into dataframes for each cell[i] 
      df = s.main_dataframe
      df_cell0 = df[df.cell.eq(0)]
      df_cell1 = df[df.cell.eq(1)]
      df_cell2 = df[df.cell.eq(2)]

      # Cell0 plot # KB better to always keep plotting separate from everything else
      # x-axis is time
      x = df_cell0["time"]
      y_0_tp = df_cell0["a_tp(Mbps)"]
      y_0_ec = df_cell0["EC(kW)"]
      y_0_ee = df_cell0["EE"]
      y_0_UEs = df_cell0["n_UEs"]

      fig, (ax0, ax1, ax2, ax3) = plt.subplots(4)

      ax0.plot(x, y_0_tp, "tab:green")
      ax1.plot(x, y_0_ec, "tab:green")
      ax2.plot(x, y_0_ee, "tab:blue")
      ax3.plot(x, y_0_UEs, "tab:red")

      ax0.set(ylabel="Throughput (Mbps)")
      ax1.set(ylabel="Power (kW)")
      ax2.set(ylabel="Energy Efficiency (Mbps/kW)")
      ax3.set(xlabel="time (s)", ylabel="Number of attached UEs")
      plt.xlim([0, max(x) + 0.5])
      plt.show()
      # print(s.main_dataframe)
