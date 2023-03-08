import os
import pandas as pd
import logging

def read_files(directory):
    """
    Reads in all TSV files in the specified directory that begin with 'KISS_07_change_random_cell_power_logfile_'
    and returns a list of DataFrames, one for each file.
    """
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.tsv') and filename.startswith('KISS_07_change_random_cell_power_logfile_'):
            logging.info(f'Reading file: {filename}')
            files.append(pd.read_csv(os.path.join(directory, filename), delimiter='\t'))
    return files

def process_dataframes(dataframes):
    """
    Takes a list of DataFrames and returns a single DataFrame with the data aggregated by "serving_cell_id".
    """
    master_df = pd.concat(dataframes)
    return master_df.groupby('serving_cell_id').agg('sum')

def main():
    # Set up logging
    logging.basicConfig(filename='example.log', level=logging.INFO)
    logging.info('Starting script.')
    
    # Define the directory where the files are located
    directory = '/path/to/directory'

    # Read in the TSV files and process them
    files = read_files(directory)
    master_df = process_dataframes(files)

    logging.info('Finished processing files.')

if __name__ == '__main__':
    main()
