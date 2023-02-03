import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from MergeTables_00 import write_df_to_feather

def read_feather(path: Path):
    df_UE = pd.read_feather(path / 'UE.fea')
    df_Cell = pd.read_feather(path / 'Cell.fea')
    df_Energy = pd.read_feather(path / 'Energy.fea')
    return df_UE, df_Cell, df_Energy



def main(path, outfile):
    df_UE, df_Cell, df_Energy = read_feather(path)

    print(df_UE.head())
    print(df_UE.shape)
    print(df_Cell.head())
    print(df_Cell.shape)
    print(df_Energy.head())
    print(df_Energy.shape)
    print('pause')

    # Finish by writing the file out
    # write_df_to_feather(df=, path=path, outfile='Edited')


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=Path, default=None,
                        help='Path with files to iterate over and merge into one pd.DataFrame (feather file).')
    parser.add_argument('-outfile', type=str, default=None, help='Outfile name.')
    args = parser.parse_args()
    main(path=args.path, outfile=args.outfile)