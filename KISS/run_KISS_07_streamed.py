import argparse
import json
import multiprocessing
import os
from pathlib import Path
import subprocess

import pandas as pd

# Define the argument parser
parser = argparse.ArgumentParser(description='Run KISS_07_change_random_cell_power.py with a range of seed values.')
parser.add_argument('-c', '--config-file', type=str, required=True, default='/Users/apw804/dev-02/EnergyModels/KISS/KISS_07_change_random_cell_power_config.json',
                    help='path to the JSON configuration file')

def process_seed(seed, config_file):
    # Load the contents of the JSON file into a dictionary
    with open(config_file) as f:
        config = json.load(f)

    # Update the "seed" value in the dictionary
    config['seed'] = seed

    # Save the updated JSON file with the seed value appended to the output filename
    out_file = Path(config_file) # Path to the output file
    new_out_file = out_file.with_stem(out_file.stem + f"_{seed}")
    config['logfile_name'] = str(new_out_file)

    with open(str(new_out_file), 'w') as f:
        json.dump(config, f, indent=4)
        # Return the path to the new config file
        return str(new_out_file)

def append_to_output_file(file_path, output_file):
    # Load the data from the file into a data frame
    df = pd.read_csv(file_path, sep='\t')

    # Append the data frame to the output file
    with open(output_file, 'a') as f:
        df.to_csv(f, header=False, index=False, sep='\t')


def run_KISS_07(config_file, output_file):
    # Call KISS_07.py with the new config file
    command = f"python KISS_07_change_random_cell_power.py -c {config_file}"
    subprocess.run(command, shell=True, check=True)

    # Append the contents of the log file to the output file
    with open(config_file) as f:
        config = json.load(f)
    seed = config['seed']
    log_file = config['logfile_name']
    append_to_output_file(log_file, output_file)
    
    print(f"Processed seed value: {seed}")
    

def main()

if __name__ == '__main__':
    # Parse the command line arguments
    args = parser.parse_args()

    # Load the contents of the JSON file into a dictionary
    with open(args.config_file) as f:
        config = json.load(f)

    # Extract the maximum seed value from the dictionary
    seed_max = config['seed']

    # Process each seed value to generate config files
    seeds = list(range(seed_max))
    new_configs = []
    output_file = 'output.tsv'
    for seed in seeds:
        new_configs.append(process_seed(seed, args.config_file))

    # Run KISS_07.py for each config file
    for config_file in new_configs:
        run_KISS_07(config_file, output_file)
        
    # Print a message to indicate that all subprocesses have completed
    print("All subprocesses completed")

    
