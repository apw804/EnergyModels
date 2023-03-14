# This script reduces the power of one randomly selected base station (cell) in the network.
# The power of the base station is reduced by 0.5 dBm at a time from 42 dBm to 30 dBm.
# The script runs kiss.py for each power value and for each seed value for a total of 1600 runs.

import argparse
import json
import multiprocessing
import time
import numpy as np

from kiss import main

def generate_config_dict_list(config_file):
    # Load the contents of the JSON file into a dictionary
    with open(config_file) as f:
        config = json.load(f)

    # Extract the maximum seed value from the dictionary
    seed_max = config['seed']

    # Extract the power_dBm value from the dictionary
    power_dBm = config['power_dBm']

    # Extract the power_dBm_end value from the dictionary
    power_dBm_end = config['power_dBm_end']

    # Extract the power_dBm_step value from the dictionary
    power_dBm_step = config['power_dBm_step']

    # Define the range of power values to test
    power_values = np.arange(power_dBm, power_dBm_end-power_dBm_step, -power_dBm_step)

    # Define the range of seed values to test 
    seeds = list(range(seed_max))

    # Define the number range of variable power cells
    n_variable_power_cells = list(range(1, config['n_variable_power_cells']+1))

    # Create a list of dictionaries
    config_dict_list = []
    for seed in seeds:
        for power_dBm in power_values:
            for variable_power_cell in n_variable_power_cells:
                # Create a new dictionary object for each iteration
                config_copy = config.copy()
                
                # Update the seed and power_dBm value in the new dictionary object
                config_copy['seed'] = seed
                config_copy['variable_cell_power_dBm'] = power_dBm
                config_copy['n_variable_power_cells'] = variable_power_cell
                
                # Append the new dictionary object to the list
                config_dict_list.append(config_copy)

    # Return the list of dictionaries
    return config_dict_list


def run_kiss(dict):
    # Run kiss.py with the dictionary as an argument
    main(dict)
    print(f'Completed: {dict["seed"]}, {dict["power_dBm"]}')
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run kiss.py against a specified config value.')
    parser.add_argument(
        '-c', 
        '--config-file', 
        type=str, 
        required=True,
        default='KISS/data/input/configs/kiss_test_00.json'
        )

    args = parser.parse_args()

    config_dict_list = generate_config_dict_list(args.config_file)
    max_processes = 8
    process_list = []

    def wait_for_processes(process_list, max_processes):
        while len(process_list) >= max_processes:
            for p in process_list:
                if not p.is_alive():
                    process_list.remove(p)
                    break
            else:
                continue
            break

    # Start the timer
    start_time = time.time()

    for config_dict in config_dict_list:
        wait_for_processes(process_list, max_processes)

        p = multiprocessing.Process(target=run_kiss, args=(config_dict,))
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()

        # Check if an exception was raised in the worker process
        if p.exitcode != 0:
            print(f"Process {p.pid} raised an exception with exit code {p.exitcode}")
            # Add error handling code here if needed

    # Stop the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"All subprocesses completed in {elapsed_time:.2f} seconds")

