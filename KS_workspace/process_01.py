# 2022-11-28
# Kishan Sthankiya
# Script to produce plots from generated simulation files
# 1 base station, 1 UE. Plot Spectral and energy efficiency

import argparse
import os
from os.path import join, isfile
from time import strftime, localtime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

cwd = str(Path.cwd())
timestamp_date = strftime('%Y-%m-%d', localtime())
timestamp_time = strftime('%H:%M:%S', localtime())


def fig_timestamp(fig, author='Kishan Sthankiya', fontsize=6, color='gray', alpha=0.7, rotation=0, prespace='  '):
    # Keith Briggs 2020-01-07
    # https://riptutorial.com/matplotlib/example/16030/coordinate-systems-and-text
    date = strftime('%Y-%m-%d %H:%M', localtime())
    fig.text(  # position text relative to Figure
        0.01, 0.005, prespace + '%s %s' % (author, date,),
        ha='left', va='bottom', fontsize=fontsize, color=color,
        rotation=rotation,
        transform=fig.transFigure, alpha=alpha)


def process(logfile: Path):
    if logfile is None:
        default_folder = cwd + '/logfiles/' + timestamp_date
        default_logfile = next(
            join(default_folder, f) for f in os.listdir(default_folder) if isfile(join(default_folder, f)))
        default_outfile_path = default_folder + '/plots/'  # FIXME actually use this!
        logfile = default_logfile

    # read the tsv logfile
    df = pd.read_csv(logfile, sep='\t')

    # sort the dataframe
    df = df.sort_values(by=['cell_id', 'time'])

    # set the index and don't drop
    df = df.set_index(keys=['cell_id'], drop=False)

    # get a list of cells
    cells = df['cell_id'].unique().tolist()

    # now we create a dataframe for each cell_id and assign to a unique dictionary key
    df_cell_dict = {}
    for i in cells:
        df_cell_dict['df_cell_' + str(i)] = df.loc[df['cell_id'] == i]

    # now we define plotting
    for df_cell in df_cell_dict.values():
        if (df_cell.n_UEs > 0).any():
            x = df_cell['cell_dBm'].sort_values(ascending=False)
            y1 = df_cell['cell_avg_se_overall']
            y2 = df_cell['cell_ee_now']
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.plot(x, y1, color='r', marker='o')
            ax.set_title(
                f'Plot for Cell {df_cell.index[0]} showing \n base station SE & EE at varying transmit power levels.')
            ax.invert_xaxis()
            ax.set_xlabel('cell power (dBm)')
            ax.set_ylabel('base station spectral efficiency (bps/Hz)', color='r', fontsize=14)

            ax2 = ax.twinx()
            ax2.plot(x, y2, color='b', marker='o')
            ax2.set_ylabel('base station energy efficiency (bps/Joule)', color='b', fontsize=14)
            fig_timestamp(fig)
            outfile_name = str(strftime('%Y-%m-%d %H:%M', localtime())) + f'_Cell{df_cell.index[0]}'
            plt.savefig(str(logfile) + f'_cell_{df_cell.index[0]}_plot.png')
            # plt.show()

    print('wait pause')

    return df_cell_dict


'''


    for each subframe
        plot the cell_dBm (x-axis) vs
            throughput for each attached UE (y1) # use different colours
                                                 # add dotted line for minimum threshold of satisfaction
            base station energy efficiency (y2)
'''

if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('-logfile', type=Path, default=None, help='full path of logfile')
    # parser.add_argument('-output', type=Path, default=Path('./test/plots'), help='directory path to write plots to')
    args = parser.parse_args()
    process(logfile=args.logfile)
