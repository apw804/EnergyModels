# Script for multiple runs of ReduceCellPower13.py
# Use as a template.
# Kishan Sthankiya
# Combine seed number and power_dBm range

import argparse
from tqdm import tqdm
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from ReduceCellPower_13 import test_01

# Set the Experiment name
name = "Exp_Seeds"


@contextmanager
def suppress_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err:  # Could add this in --> "redirect_stdout(fnull) as out:"
            yield (err)


if __name__ == '__main__':
    for i in tqdm(range(10)):
        for j in tqdm(range(30, 50)):
            with suppress_stderr():
                parser = argparse.ArgumentParser()
                parser.add_argument('-seed', type=int, default=i, help='seed value for random number generator')
                parser.add_argument('-isd', type=float, default=500.0,
                                    help='Base station inter-site distance in metres')
                parser.add_argument('-sim_radius', type=float, default=1000.0,
                                    help='Simulation bounds radius in metres')
                parser.add_argument('-nues', type=int, default=1000, help='number of UEs')
                parser.add_argument('-subbands', type=int, default=1, help='number of subbands')
                parser.add_argument('-power_dBm', type=float, default=j,
                                    help='set the transmit power of the cell in dBm')
                parser.add_argument('-until', type=float, default=2.0, help='simulation time')
                parser.add_argument('-logging_interval', type=float, default=1.0,
                                    help='Sampling interval (seconds) for simulation data capture + UEs reports sending.')
                parser.add_argument('-experiment_name', type=str, default=None,
                                    help='name of a specific experiment to influence the output log names.')
                args = parser.parse_args()

                test_01(seed=args.seed, subbands=args.subbands, isd=args.isd, sim_radius=args.sim_radius,
                        power_dBm=args.power_dBm, nues=args.nues, until=args.until, sim_args_dict=args.__dict__,
                        logging_interval=args.logging_interval, experiment_name=name)
