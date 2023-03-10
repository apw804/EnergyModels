import argparse
from pathlib import Path
import pandas as pd

from logging_KISS_07 import get_logger

# Get current script name and current working directory
script_name = Path(__file__).stem
cwd = Path.cwd()

# Create a logger object
logger = get_logger(script_name, cwd)

def analyze_data(data_file):
    # Read in the data from the feather file
    logger.info(f'Reading data from {data_file}...')
    df = pd.read_feather(data_file)
    df = df.reset_index(drop=True)

    # Define the columns and metrics to analyze
    columns = ['cell_throughput(Mb/s)', 'cell_power(kW)', 'cell_ee(bits/J)', 'cell_se(bits/Hz)']
    metrics = ['mean', 'std']

    # Perform your data analysis here
    logger.info('Starting data analysis...')

    # Calculate the mean and standard deviation for each column
    results = []
    for col in columns:
        col_stats = df.groupby(['serving_cell_id', 'sc_power(dBm)'])[col].agg(metrics)
        col_stats.columns = [f'{col}_{metric}' for metric in metrics]
        results.append(col_stats)

    # Combine the results into a single DataFrame
    stats_df = pd.concat(results, axis=1)

    logger.info('Data analysis complete.')
    return stats_df


def main(data_file_path):
    # Call the analyze_data function with the specified data file
    analyze_data(data_file_path)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_file_path', 
        type=str, 
        help='Path to feather file',
        default='/Users/apw804/dev-02/EnergyModels/KISS/KISS_07_results/KISS_07_results_2023-03-09T22:22:03.feather'
        )
    args = parser.parse_args()

    # Call the main function
    main(data_file_path=args.data_file_path)
