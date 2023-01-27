# Script for multiple runs of ReduceCellPower13.py
# Use as a template.
# Kishan Sthankiya
# Combine seed number and power_dBm range

import argparse

import numpy as np
from tqdm import tqdm
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from ReduceCellPower_13 import main

# Set the Experiment name
name = "n2_f1960MHz_FDD_bw10MHz"

# Set the starting power_dBm for all cells
p_start: float = 30.0

# Set the target power_dBm for all cells
p_end: float = 49.0

# Calculate the `until` time
delta_p = p_end - p_start
until = np.abs(delta_p) + 2

@contextmanager
def suppress_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


if __name__ == '__main__':
    for i in tqdm(range(100)):
            with suppress_stderr():
                parser = argparse.ArgumentParser()
                parser.add_argument('-seed', type=int, default=i, help='seed value for random number generator')
                parser.add_argument('-isd', type=float, default=500.0,
                                    help='Base station inter-site distance in metres')
                parser.add_argument('-sim_radius', type=float, default=1000.0,
                                    help='Simulation bounds radius in metres')
                parser.add_argument('-nues', type=int, default=1000, help='number of UEs')
                parser.add_argument('-subbands', type=int, default=1, help='number of subbands')
                parser.add_argument('-fc_GHz', type=float, default=3.40, help='Centre frequency in GHz')
                parser.add_argument('-h_UT', type=float, default=1.5,
                                    help='Height of User Terminal (=UE) in metres (default=1.5)')
                parser.add_argument('-h_BS', type=float, default=25.0,
                                    help='Height of Base Station in metres (default=25)')
                parser.add_argument('-power_dBm', type=float, default=p_start,
                                    help='set the transmit power of the cell in dBm')
                parser.add_argument('-until', type=float, default=until, help='simulation time')
                parser.add_argument('-logging_interval', type=float, default=1.0,
                                    help='Sampling interval (seconds) for simulation data capture + UEs reports sending.')
                parser.add_argument('-experiment_name', type=str, default=None,
                                    help='name of a specific experiment to influence the output log names.')
                parser.add_argument('-target_power_dBm', type=float, default=p_end,
                                    help='the target power to reach from the initial power set.')
                args = parser.parse_args()

                main(seed=args.seed, subbands=args.subbands,
                     isd=args.isd, sim_radius=args.sim_radius,
                     power_dBm=args.power_dBm, nues=args.nues,
                     fc_GHz=args.fc_GHz, h_UT=args.h_UT, h_BS=args.h_BS,
                     until=args.until, sim_args_dict=args.__dict__,
                     logging_interval=args.logging_interval,
                     experiment_name=args.experiment_name,
                     target_power_dBm=args.target_power_dBm)
