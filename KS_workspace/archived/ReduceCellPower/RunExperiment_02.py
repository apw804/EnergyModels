# Script for multiple runs of ReduceCellPower13.py
# Use as a template.
# Kishan Sthankiya
# Combine seed number and power_dBm range

import json
import subprocess
from pathlib import Path

from tqdm import tqdm


def run_sim(json_file, sim_file):

    with open(json_file) as f:
        data = json.load(f)

    seeds = data.pop("seeds")
    start, end = seeds["start"], seeds["end"]

    for seed in tqdm(range(start, end), desc=f"Running {sim_file} for {len(seeds) + 1} seed values."):
        args = ["python", sim_file]
        for key, value in data.items():
            args.append(f"-{key}")
            args.append(str(value))
        args.append(f"-seed")
        args.append(str(seed))
        results = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        with open('output_error.txt', 'w') as file:
            result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            file.write(result.stdout.decode())
            file.write(result.stderr.decode())


if __name__ == "__main__":
    run_sim("RunExperiment_02_args.json", "ReduceCellPower_14.py")
