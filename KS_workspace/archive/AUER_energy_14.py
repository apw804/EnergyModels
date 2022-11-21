# Keith Briggs 2022-10-14 16:06 - note code suggestions here!
# Kishan Sthankiya
# 2022-10-12 16:39:48 AIMM_core_simulator_17 changes
# Example: python3 power_model001.py -ncells=1 -nues=1 -until=100
import argparse
from dataclasses import dataclass
from datetime import datetime
from os import getcwd
from pathlib import Path
from sys import path

path.append(f'{str(Path.home())}/AIMM_simulator-1.0/')
from time import localtime, strftime, strptime, time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from AIMM_simulator import (UE, Cell, Logger, Scenario, Sim, from_dB,
                                    to_dB)
from NR_5G_standard_functions_00 import Radio_state

rng = np.random.default_rng(seed=0) # FIXME

@dataclass(frozen=True)
class SmallCellParameters:
  ''' Object for setting small cell base station parameters.'''
  eta_PA:float=              0.067
  power_RF_watts:float=      1.0
  power_baseband_watts:float=3.0
  loss_feed_dB:float=        0.00
  loss_DC_dB:float=          0.09
  loss_cool_dB:float=        0.00
  loss_mains_dB:float=       0.11

@dataclass(frozen=True)
class MacroCellParameters:
  ''' Object for setting macro cell base station parameters.'''
  eta_pa:float=               0.067
  power_rf_watts:float=      12.9
  power_baseband_watts:float=29.6
  loss_feed_db:float=        -3.0
  loss_dc_db:float=           0.075
  loss_cool_db:float=         0.10
  loss_mains_db:float=        0.09


class PowerModel:
  '''
  Power model for a base station.
  Based on EARTH framework (10.1109/MWC.2011.6056691).
  '''

  def __init__(self, sim):
    '''
    Initialisation variables used for the base station power model.
    '''
    self.sim=sim
    self.cell=Cell(sim)
    if self.cell.power_dBm < 30:
      self.params = SmallCellParameters()
    else:
      self.params = MacroCellParameters()
    if self.cell.pattern is None:
      self.antennas=3  # Assume base station is 3 antennas of 120 degrees
    else:
      if isinstance(cell.pattern, np.array):
          self.antennas=1  # If there's an array assume that it's 1 antenna
      else:
          self.antennas=1  # assuming the function is unidirectional (for now)
    self.sectors=self.antennas
    print('pause')
    self.loss_feeder_ratio=from_dB(self.params.loss_feed_db)

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
    tx_BS_dBm=cell.power_dBm
    carriers=cell.n_subbands


    def power_pa_watts(self, tx_BS_dBm):
      self.tx_max_BS_watts = from_dB(tx_BS_dBm)
      return self.tx_max_BS_watts/(self.params.eta_pa*(1-self.loss_feeder_ratio))

    n_trx=carriers*self.antennas*self.sectors
    power_bs_sum_watts=power_pa_watts(self, tx_BS_dBm)+self.params.power_rf_watts+self.params.power_baseband_watts
    power_bs_loss=(1-self.params.loss_dc_db)*(1-self.params.loss_mains_db)*(1-self.params.loss_cool_db)
    power_bs_frac=power_bs_sum_watts/power_bs_loss
    power_bs_var=n_trx*power_bs_frac
    power_bs_static=n_trx*(power_pa_watts(self,0)+self.params.power_rf_watts+self.params.power_baseband_watts/power_bs_loss)
    return cell.interval*(power_bs_static+power_bs_var)

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

        ec = (PowerModel.cell_energy(cell=cell.i) / 1000)  # divide by 1000 to convert from watts to kilowatts
        # Calculate the energy efficiency
        if tp == 0.0:
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
    filename = str(Path(__file__).stem+'_log_'+str(datetime.now(tz=None)))
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
        ue.xyz[:2] += 20 * rng.random()
        yield s.sim.wait(interval)
# END class QmScenario

def em001(ncells=1,nues=1,until=10):

  '''Define the parameters to set up and execute the simulation.
  '''
  interval=1.0e0 # KB cell interval - other intervals will be scaled to this
  sim = Sim()
  power_model = PowerModel(sim)
  for i in range(ncells):
    sim.make_cell(interval=interval,verbosity=1)
  for i in range(nues):
    ue = sim.make_UE(verbosity=1)
    ue.interval=0.1*interval # FIXME explain interval scaling
    ue.attach_to_nearest_cell()
  for cell in sim.cells:
    cell.set_f_callback(power_model.f_callback(x=cell))
 
  sim.add_logger(QmEnergyLogger(sim, logging_interval=1*interval))
  scenario=QmScenario(sim,verbosity=1)
  sim.add_scenario(scenario) # FIXME interval?
  sim.run(until=until)
  print(f'reference_signal_power={power_model.rsp_dBm(cell):.2f}dBm')

if __name__=='__main__': # a simple self-test
  np.set_printoptions(precision=4,linewidth=200)
  parser=argparse.ArgumentParser()
  parser.add_argument('-ncells',type=int,default= 4,help='number of cells')
  parser.add_argument('-nues',  type=int,default=10,help='number of UEs')
  parser.add_argument('-until', type=float,default=100.0,help='simulation time')
  args=parser.parse_args()
  em001(ncells=args.ncells,nues=args.nues,until=args.until)
