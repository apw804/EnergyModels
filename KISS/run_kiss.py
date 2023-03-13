import argparse
import json
import multiprocessing
import os
from pathlib import Path
import subprocess

import numpy as np

# Define the argument parser
parser = argparse.ArgumentParser(
    description='Run KISS_07_change_random_cell_power.py with a range of seed values.')
parser.add_argument('-c', '--config-file', type=str, required=True,
                    default='/Users/apw804/dev-02/EnergyModels/KISS/data/input/kiss_07/KISS_07_change_random_cell_power_config.json')


def process_args(seed, config_file):
    # FIXME - add power_dBm update here
    # Load the contents of the JSON file into a dictionary
    with open(config_file) as f:
        config = json.load(f)

    # Update the "seed" and  value in the dictionary
    config['seed'] = seed

    # Save the updated JSON file with the seed value appended to the output filename
    out_file = Path(config_file)  # Path to the output file
    new_out_file = out_file.with_stem(out_file.stem + f"_{seed}")
    config['logfile_name'] = str(new_out_file)

    with open(str(new_out_file), 'w') as f:

        json.dump(config, f, indent=4)
        # Return the path to the new config file
        return str(new_out_file)


def run_KISS_07(config_file):
    # Call KISS_07.py with the new config file
    command = f"python KISS_07_change_random_cell_power.py -c {config_file}"
    subprocess.run(command, shell=True, check=True)
    with open(config_file) as f:
        config = json.load(f)
    seed = config['seed']
    print(f"Processed seed value: {seed}")
    


if __name__ == '__main__':
    # Parse the command line arguments
    args = parser.parse_args()

    # Load the contents of the JSON file into a dictionary
    with open(args.config_file) as f:
        config = json.load(f)

    # Extract the maximum seed value from the dictionary
    seed_max = config['seed']

    # Extract the power_dBm value from the dictionary
    power_dBm = config['power_dBm']
    # Extract the new_power_dBm value from the dictionary
    new_power_dBm = config['new_power_dBm']
    # Define the range of power values to test
    power_values = np.arange(power_dBm, new_power_dBm, 0.5)

    # FIXME - need to set the json power dBm value to the new value

    # Process each seed value to generate config files
    seeds = list(range(seed_max))
    new_configs = []
    for seed in seeds:
        new_configs.append(process_args(seed, args.config_file))

    # Run KISS_07.py for each config file
    for config_file in new_configs:
        run_KISS_07(config_file)

    # Print a message to indicate that all subprocesses have completed
    print("All subprocesses completed")
