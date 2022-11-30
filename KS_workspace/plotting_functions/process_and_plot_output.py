# 2022-11-28
# Kishan Sthankiya
# Script to produce plots from generated simulation files
import argparse
import os
from os.path import join, isfile

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

cwd = str(Path.cwd())
test_folder = cwd + '/test'
default_logfile = next(join(test_folder, f) for f in os.listdir(test_folder) if isfile(join(test_folder, f)))

print(default_logfile)


def process(logfile: Path = default_logfile):
    # read the tsv logfile
    df = pd.read_csv(logfile, sep='\t')

    # sort the dataframe
    df.sort_values(by=['cell_id', 'time'])

    # set the index and don't drop
    df.set_index(keys=['cell_id'], drop=False)

    # get a list of cells
    cells = df['cell_id'].unique().tolist()

    # now we create a dataframe for each cell_id and assign to a unique global variable
    for i in cells:
        globals()[f'df_cell_{i}'] = df.loc[df['cell_id'] == i]


    def cell_subframe(dataset):
        df_cell = []
        for df in dataset:
            pass

    return print(globals())


'''
    for each dataframe: # which is a different seed value
        get the cell_id column
        for each unique cell_id
            split the dataframe into subframes

    for each subframe
        plot the cell_dBm (x-axis) vs
            throughput for each attached UE (y1) # use different colours
                                                 # add dotted line for minimum threshold of satisfaction
            base station energy efficiency (y2)
'''

if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('-logfile', type=Path, default=default_logfile, help='full path of logfile')
    # parser.add_argument('-output', type=Path, default=Path('./test/plots'), help='directory path to write plots to')
    args = parser.parse_args()
    process(logfile=args.logfile)
