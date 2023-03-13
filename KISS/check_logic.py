import json

import numpy as np

config_file = '/Users/apw804/dev-02/EnergyModels/KISS/data/input/configs/kiss_test_00.json'

def main(config_file):
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

    # Create a list of dictionaries
    config_dict_list = []
    for seed in seeds:
        for power_dBm in power_values:
            # Create a new dictionary object for each iteration
            config_copy = config.copy()
            
            # Update the seed and power_dBm value in the new dictionary object
            config_copy['seed'] = seed
            config_copy['power_dBm'] = power_dBm
            
            # Append the new dictionary object to the list
            config_dict_list.append(config_copy)

    # Return the list of dictionaries
    return config_dict_list


if __name__ == '__main__':
    main(config_file)