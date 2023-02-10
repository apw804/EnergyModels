import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def import_tsv_files(path: Path):
    all_files = list(path.iterdir())
    df = pd.concat((pd.read_csv(f, sep='\t') for f in all_files), ignore_index=True)
    return df


def write_df_to_feather(df: pd.DataFrame, path: Path, outfile: str = None):
    if outfile is None:
        outpath =  str(path) + '/' + str(Path(path).resolve().stem) + '.fea'
    else:
        outpath = str(path) + '/' + outfile + '.fea'
    pd.DataFrame.to_feather(df, outpath)
    print(f'DataFrame written to {outpath}.')

def write_df_to_csv(df: pd.DataFrame, path: Path, outfile: str = None):
    if outfile is None:
        outpath =  str(path) + '/' + str(Path(path).resolve().stem) + '.csv'
    else:
        outpath = str(path) + '/' + outfile + '.csv'
    pd.DataFrame.to_csv(df, outpath)
    print(f'DataFrame written to {outpath}.')


def main(path, outfile, type):
    temp_df = import_tsv_files(path)
    temp_df.head()
    if type == 'csv':
        write_df_to_csv(temp_df, path=path)
    else:
        write_df_to_feather(temp_df, path)


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=Path, default=None,
                        help='Path with files to iterate over and merge into one pd.DataFrame (feather file).')
    parser.add_argument('-outfile', type=str, default=None, help='Outfile name.')
    parser.add_argument('-type', type=str, default=None, help='output file type (csv or feather).')
    args = parser.parse_args()
    
    main(path=args.path, outfile=args.outfile, type=args.type)
