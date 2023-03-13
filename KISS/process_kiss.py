import argparse
import os
from pathlib import Path

import pandas as pd

from logging_kiss import get_logger
from utils_kiss import get_timestamp

def read_files(directory, file_starts_with):
    """
    Read all TSV files in a directory that begin with a specified string and 
    return a list of DataFrames.
    """
    files = [pd.read_csv(os.path.join(directory, filename), delimiter='\t')
             for filename in os.listdir(directory)
             if filename.endswith('.tsv') and 
                filename.startswith(file_starts_with)]
    logger.info(f'Read {len(files)} files.')
    return files


def concat_dataframes(dataframes):
    """
    Concatenate a list of DataFrames and return a single DataFrame.
    """
    logger.info('Concatenating dataframes.')
    master_df = pd.concat(dataframes, ignore_index=True)
    logger.info('Sorting data by seed, serving_cell_id, and ue_id.')
    master_df = master_df.sort_values(['seed', 'serving_cell_id', 'ue_id'])
    logger.info('Finished processing dataframes.')
    return master_df


def write_dataframe(df: pd.DataFrame, path: Path, outfile: str = None,
                    file_type: str = 'feather'):
    """
    Write a Pandas DataFrame to a file.
    """
    if outfile is None:
        outfile = path.resolve().stem
    outpath = path / f'{outfile}_{get_timestamp()}.{file_type}'
    if file_type == 'feather':
        df = df.reset_index()
        df.to_feather(outpath)
    elif file_type == 'csv':
        df.to_csv(outpath, index=False)
    else:
        raise ValueError(f'Invalid file type: {file_type}')
    logger.info(f'DataFrame written to {outpath}.')


def main(directory, file_starts_with, logging_enabled=True, 
         outfile=None, file_type='csv'):

    # Set up logging
    if logging_enabled:
        
        logger.info('Starting script.')
        logger.info(f'Processing files in directory: {directory}')
        logger.info(f'Output file type: {file_type}')
        logger.info(f'Output file name: {outfile}')

    # Read in the TSV files and process them
    files = read_files(directory, file_starts_with)
    master_df = concat_dataframes(files)

    # Write the output file
    write_dataframe(master_df, Path(directory), outfile, file_type)

    if logging_enabled:
        logger.info('Finished processing files.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--no-logging', 
        action='store_false', 
        help='Turn off logging'
    )
    parser.add_argument(
        '--dir', 
        type=str, 
        help='Directory where files are located', 
        default='/Users/apw804/dev-02/EnergyModels/KISS/data/output/kiss_07/2023-03-13'
    )
    parser.add_argument(
        '--file_starts_with', 
        type=str, 
        help='String that all files start with', 
        default='KISS_07_change_random_cell_power'
    )
    parser.add_argument(
        '--outfile', 
        type=str, 
        help='Name of output file'
    )
    parser.add_argument(
        '--file_type', 
        type=str, 
        help='File type of output file (csv or feather)', 
        default='csv'
    )
    args = parser.parse_args()

    script_name = Path(__file__).resolve().stem
    logger = get_logger(script_name, args.dir)

    main(
        directory=args.dir, 
        file_starts_with=args.file_starts_with, 
        logging_enabled=args.no_logging, 
        outfile=args.outfile, 
        file_type=args.file_type
    )



