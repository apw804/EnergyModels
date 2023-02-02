# Keith Briggs 2022-10-14 16:06 - note code suggestions here!
# Kishan Sthankiya
# 2022-10-12 16:39:48 Basic script to check understanding of AIMM_core_simulator_17.py changes
# Example: python3 em001.py -ncells=1 -nues=1 -until=100

import argparse
from dataclasses import dataclass
from os import getcwd
from pathlib import Path
from sys import path
from time import localtime, strftime, strptime, time

path.append(f'{str(Path.home())}/AIMM_simulator-1.0/')
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from AIMM_simulator_core_17 import (UE, Cell, Logger, Scenario, Sim, from_dB,
                                    to_dB)
from NR_5G_standard_functions_00 import Radio_state
from numpy.random import seed, standard_normal  # OLD!

seed(0) # FIXME

def fig_timestamp(fig,fontsize=6,color='black',alpha=0.7,rotation=0,prespace='  ',author=''):
  # add an author and date-time to bottom of plot
  date=strftime('%Y-%m-%d %H:%M',localtime())
  fig.text( # position text relative to Figure
    0.01,0.005,prespace+f'{author} {date}',
    ha='left',va='bottom',fontsize=fontsize,color=color,
    rotation=rotation, transform=fig.transFigure,alpha=alpha)

@dataclass
class MacroCellParameters:
  ''' Object for setting macro cell base station parameters.'''
  eta_PA:float=               0.067
  power_RF_watts:float=      12.9
  power_baseband_watts:float=29.6
  loss_feed_dB:float=        -3.0
  loss_DC_dB:float=           0.075
  loss_cool_dB:float=         0.10
  loss_mains_dB:float=        0.09

@dataclass
class SmallCellParameters:
  ''' Object for setting small cell base station parameters.'''
  eta_PA:float=              0.067
  power_RF_watts:float=      1.0
  power_baseband_watts:float=3.0
  loss_feed_dB:float=        0.00
  loss_DC_dB:float=          0.09
  loss_cool_dB:float=        0.00
  loss_mains_dB:float=       0.11

class QM_Energy_001:
  ''' Initial energy model based on EARTH framework for a macro base station.'''

  def __init__(self, sim):
    '''
    Variable initialisation.
    '''
    self.sim=sim
    if cell.power_dBm>=30:
      self.cell_params=MacroCellParameters()
    else:
      self.cell_params=SmallCellParameters()

  def rsp_dBm(self, cell):
    '''
    Calculates the Reference Signal Power in dBm. Based on maximum number of physical resource blocks in the current radio state.
    '''
    max_rbs=Radio_state.nPRB
    return cell.power_dBm-to_dB(max_rbs*12)  # 12 is the number of resource elements in a PRB

  def cell_energy(self, cell):
    '''
    EARTH framework energy model for a base station.
    '''
    carriers=cell.n_subbands

    # FIXME move all invariant stuff to __init__
    if cell.pattern is None:
        # Assume base station is 3 antennas of 120 degrees
        antennas=3
    else:
        # If there's an array assume that it's 1 antenna
        if isinstance(cell.pattern, np.array):
            antennas=1
        else:
            antennas=1  # assuming the function is unidirectional (for now)
    sectors=antennas
    # Convert feeder loss to watts
    loss_feeder_watts=from_dB(-3.0)   # Reference value from DOI:10.1109/MWC.2011.6056691
    tx_max_BS_dBm=cell.power_dBm
    # Set values if it's a macro cell
    if tx_max_BS_dBm>=30:
        eta_PA=0.311
        power_RF_watts=12.9
        power_baseband_watts=29.6
        loss_DC=0.075
        loss_cool=0.1
        loss_mains=0.09
    # Set values for small cell
    else:
        eta_PA=0.067
        loss_feeder_watts=0.00
        power_RF_watts=1
        power_baseband_watts=3
        loss_DC=0.09
        loss_cool=0.00
        loss_mains=0.11
    # better way:
    @dataclass
    class Small_cell_parameters:
      eta_PA=              0.067
      loss_feeder_watts=   0.00
      power_RF_watts=      1.0
      power_baseband_watts=3.0
      loss_DC=             0.09
      loss_cool=           0.00
      loss_mains=          0.11
    # and then refer to Small_cell_parameters.eta_PA etc.

    def power_PA_watts(self, tx_BS_dBm):
      self.tx_max_BS_watts = from_dB(tx_BS_dBm)
      return self.tx_max_BS_watts/(eta_PA*(1-loss_feeder_watts))

    n_trx=carriers*antennas*sectors
    power_BS_sum_watts=power_PA_watts(self, tx_max_BS_dBm)+power_RF_watts+power_baseband_watts
    power_BS_loss=(1-loss_DC)*(1-loss_mains)*(1 - loss_cool)
    power_BS_frac=power_BS_sum_watts/power_BS_loss
    power_BS_var=n_trx*power_BS_frac
    power_BS_static=n_trx*(power_PA_watts(self,0)+power_RF_watts+power_baseband_watts/power_BS_loss)
    return cell.interval*(power_BS_static+power_BS_var)

  def f_callback(self,x,**kwargs):
    #print(kwargs)
    if isinstance(x,Cell):
      self.cell_energy(x)
    elif isinstance(x,UE):
      self.ue_energy(x)
# END class Energy

class QmEnergyLogger(Logger):
  '''
  Custom Logger for the em001 energy model.
  '''

  # Constructor for QmEnergyLogger with additional parameter
  def __init__(s, *args, **kwargs):
    # Calling the parent constructor using ClassName.__init__()
    Logger.__init__(s, *args, **kwargs)
    # Adding the cols and main_dataframe
    s.cols = (
      'time',
      'cell',
      'n_UEs',
      'tp_Mbps',
      'EC(kW)',
      'EE',
    )
    s.main_dataframe = pd.DataFrame(data=None, columns=list(s.cols))

  def append_row(s, new_row):
    temp_df = pd.DataFrame(data=[new_row], columns=list(s.cols))
    s.main_dataframe = pd.concat([s.main_dataframe, temp_df])

  def loop(s):
    # Write to stdout
    yield s.sim.wait(s.logging_interval)
    s.f.write("#time\tcell\tn_UEs\ttp_Mbps\tEC (kW)\tEE (Mbps/kW)\n")
    while True:
      # Needs to be per cell in the simulator
      for cell in s.sim.cells:
        # Get timestamp
        tm = s.sim.env.now
        # Get cell ID
        #cell_i = cell.i
        # Get number of UEs attached to this cell
        #n_UEs = len(cell.attached)
        n_UEs = cell.get_nattached()
        # Get total throughput of attached UEs
        tp = 0.0
        #for ue_i in s.sim.cells[cell_i].attached:
        #  #tp += s.sim.cells[cell_i].get_UE_throughput(ue_i)
        #  # possibly better...
        #  ue=s.sim.UEs[ue_i]
        #  tp += cell.get_UE_throughput(ue.i)
        # or even better...
        tp=sum(cell.get_UE_throughput(ue_i) for ue_i in cell.attached)

        ec = (QM_Energy_001.cell_energy(s, cell) / 1000)  # divide by 1000 to convert from watts to kilowatts
        # Calculate the energy efficiency
        if tp == 0:
          ee = 0.0  # KB think about types - should this be 0.0?
        else:
          ee = tp / ec

        # Write to stdout
        s.f.write(
          f"{tm:10.2f}\t{cell.i:2}\t{n_UEs:2}\t{tp:10.2f}\t{ec:10.2f}\t{ee:10.2f}\n"
        )

        # Write these variables to the main_dataframe
        row = (tm, cell.i, n_UEs, tp, ec, ee)
        new_row = [np.round(each_arr, decimals=2) for each_arr in row]
        s.append_row(new_row)

      yield s.sim.wait(s.logging_interval)

  def plot(s):

    # Separate out into dataframes for each cell[i]
      df = s.main_dataframe
      df_cell0 = df[df.cell.eq(0)]
      df_cell1 = df[df.cell.eq(1)]
      df_cell2 = df[df.cell.eq(2)]

    # Cell0 plot
      # x-axis is time
      x = df_cell0["time"]
      y_0_tp = df_cell0["tp_Mbps"]
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
      fig_timestamp(fig,rotation=0,fontsize=6,author='SK')
      fig.savefig('foo.png')
      #plt.show()
      # print(s.main_dataframe)

  def finalize(s):
    cwd=getcwd()
    filename = str(Path(__file__).stem+'log')
    f=open(filename,'w')

    #x_formatted=f'{x:6g}'.ljust(7,'0') # 6 significant figures, left-justified and right-padded with zeros
    #f'{x_formatted}'
    s.main_dataframe.style.format('{>0.2f}')
    s.main_dataframe.to_csv(f, sep='\t', index=False, encoding='ascii')
    # f.close()
    s.plot()
# END class QmEnergyLogger

class QmScenario(Scenario):
  # This loop sets the amount of time between each event
  def loop(s, interval=0.1):
    while True:
      for ue in s.sim.UEs:
        # Here we update the position of the UEs in each iteration of the loop ()
        ue.xyz[:2] += 20 * standard_normal(2)
        yield s.sim.wait(interval)
# END class QmScenario

def em001(ncells=1,nues=1,until=10):
  '''Define the parameters to set up and execute the simulation.
  '''
  interval=1.0e0 # KB cell interval - other intervals will be scaled to this
  sim = Sim()
  for i in range(ncells):
    sim.make_cell(interval=interval,verbosity=1)
  for i in range(nues):
    ue = sim.make_UE(verbosity=1)
    ue.interval=0.1*interval # FIXME explain interval scaling
    ue.attach_to_nearest_cell()
  em=QM_Energy_001(sim)
  for cell in sim.cells:
    cell.set_f_callback(em.f_callback, cell_i=cell.i)
  sim.add_logger(QmEnergyLogger(sim, logging_interval=1*interval))
  scenario=QmScenario(sim,verbosity=1)
  sim.add_scenario(scenario) # FIXME interval?
  sim.run(until=until)
  print(f'reference_signal_power={em.rsp_dBm(cell):.2f}dBm')

if __name__=='__main__': # a simple self-test
  np.set_printoptions(precision=4,linewidth=200)
  parser=argparse.ArgumentParser()
  parser.add_argument('-ncells',type=int,default= 4,help='number of cells')
  parser.add_argument('-nues',  type=int,default=10,help='number of UEs')
  parser.add_argument('-until', type=float,default=100.0,help='simulation time')
  args=parser.parse_args()
  em001(ncells=args.ncells,nues=args.nues,until=args.until)
