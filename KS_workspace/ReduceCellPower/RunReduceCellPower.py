# Script for multiple runs of ReduceCellPower12.py
# Kishan Sthankiya
import argparse
import numpy as np

from ReduceCellPower_12 import test_01

# Define the arguments
seeds_list = [i for i in range(3)]

if __name__ == '__main__':
    # Set default values
    np.set_printoptions(precision=4, linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=0, help='seed value for random number generator')
    parser.add_argument('-isd', type=float, default=500.0, help='Base station inter-site distance in metres')
    parser.add_argument('-sim_radius', type=float, default=1000.0, help='Simulation bounds radius in metres')
    parser.add_argument('-nues', type=int, default=10, help='number of UEs')
    parser.add_argument('-subbands', type=int, default=1, help='number of subbands')
    parser.add_argument('-power_dBm', type=float, default=43.0, help='set the transmit power of the cell in dBm')
    parser.add_argument('-until', type=float, default=2.0, help='simulation time')
    parser.add_argument('-logging_interval', type=float, default=1.0,
                        help='Logging interval (in seconds) for the functions that will capture simulation data and for how often the UEs will send reports to their cells.')
    args = parser.parse_args()

    for i in seeds_list:
        test_01(seed=args.seed, subbands=args.subbands, isd=args.isd, sim_radius=args.sim_radius,
                power_dBm=args.power_dBm, nues=args.nues, until=args.until, sim_args_dict=args.__dict__,
                logging_interval=args.logging_interval)
