# KISS (Keep It Simple Stupid!) v2
import argparse, json
from pathlib import Path
from sys import stdout
from datetime import datetime
from time import localtime, strftime
import numpy as np
from AIMM_simulator import Sim, Logger, np_array_to_str
from hexalattice.hexalattice import *
import matplotlib.pyplot as plt
import pandas as pd


# Set variables for later use
script_name = Path(__file__).stem
timestamp = datetime.now()
timestamp_iso = timestamp.isoformat(timespec='seconds')
logfile = '_'.join([script_name, 'logfile', timestamp_iso])+'.tsv'


class MyLogger(Logger):

    def __init__(self, sim, func=None, header='', f=stdout, logging_interval=1.0, np_array_to_str=np_array_to_str):
        self.dataframe = None
        super(MyLogger, self).__init__(sim, func=None, header='',
              f=stdout, logging_interval=1.0, np_array_to_str=np_array_to_str)

    def check_type_format(self, value):
        if value is None or value == float("-inf"):
            return np.nan
        return value

    def get_data(self):
        # Create an empty list to store generated data
        data = []
        # Keep a list of column names to track
        columns = ["time", "serving_cell_id", "ue_id",
            "distance_to_cell(m)", "throughput(Mb/s)", "rsrp", "sinr(dB)", "cqi"]
        for cell in self.sim.cells:
               for attached_ue_id in cell.attached:
                    UE = self.sim.UEs[attached_ue_id]
                    serving_cell = UE.serving_cell
                    tm = self.sim.env.now                                   # current time
                    sc_id = serving_cell.i                                  # current UE serving_cel
                    sc_xy = serving_cell.get_xyz()[:2]                      # current UE serving_cell xy position
                    ue_id = UE.i                                            # current UE ID
                    ue_xy = UE.get_xyz()[:2]                                # current UE xy position
                    d2sc = np.linalg.norm(sc_xy - ue_xy)                    # current UE distance to serving_cell
                    tp = serving_cell.get_UE_throughput(attached_ue_id)     # current UE throughput
                    rp = serving_cell.get_RSRP_reports_dict()[ue_id]        # current UE rsrp from serving_cell
                    sinr = UE.sinr_dB                                       # current UE sinr from serving_cell
                    cqi = UE.cqi                                            # current UE cqi from serving_cell

                    # Get the above as a list
                    data_list = [tm, sc_id, ue_id, d2sc, tp, rp, sinr, cqi]

                    # Write above to `data` list
                    data.append(data_list)
        return columns, data

    def add_to_dataframe(self, col_labels, new_data, ignore_index):
        if self.dataframe is None:
            self.dataframe = pd.DataFrame(new_data, columns=col_labels)
        else:
            new_data_df = pd.DataFrame(data=new_data, columns=col_labels)
            self.dataframe = pd.concat(
                [self.dataframe, new_data_df], verify_integrity=True, ignore_index=ignore_index)

    def run_routine(self, ignore_index=False):
        col_labels, new_data = self.get_data()
        self.add_to_dataframe(col_labels=col_labels,
                              new_data=new_data, ignore_index=ignore_index)

    def loop(self):
        """Main loop for logger."""
        while True:
            self.run_routine()
            print(f'logger time={self.sim.env.now}')
            yield self.sim.wait(self.logging_interval)

    def finalize(self):
        '''
        Function called at end of simulation, to implement any required finalization actions.
        '''
        print(f'Finalize time={self.sim.env.now}')

        # Run routine for final time step
        self.run_routine(ignore_index=True)

        # Create a copy of the final DataFrame
        df = self.dataframe.copy()

        # Set a dictionary column datatypes and format
        df_dtypes_format = {
            "time": (float, "{:.0f}"),
            "serving_cell_id": (int, "{:.0f}"),
            "ue_id": (int, "{:.0f}"),
            "distance_to_cell(m)": (float, "{:.2f}"),
            "throughput(Mb/s)": (float, "{:.2f}"),
            "rsrp": (float, "{:.0f}"),
            "sinr(dB)": (float, "{:.2f}")}

        # Apply the data types
        for col, (dtype, format_str) in df_dtypes_format.items():
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(value=np.nan, inplace=True)
            df[col] = df[col].astype(dtype).round(3)
            df[col].format = format_str

        # Apply a style to the df DataFrame and print to screen
        df.style.format(
            {col: format_str for col, (dtype, format_str) in df_dtypes_format.items()})
        print(df)

        # Write the MyLogger dataframe to TSV file
        df.to_csv(logfile, sep="\t", index=False)

# END MyLogger class


def generate_ppp_points(sim, expected_pts=100, sim_radius=500.0):
    """
    Generates npts points, distributed according to a homogeneous PPP
    with intensity lamb and returns an array of distances to the origin.
    """
    sim_rng = sim.rng

    # Simulation window parameters
    sim_radius = sim_radius  # radius of disk
    xx0 = 0
    yy0 = 0  # centre of disk
    areaTotal = np.pi * sim_radius ** 2  # area of disk

    # Point process parameters
    # intensity (ie mean density) of the Poisson process
    lambda0 = expected_pts / areaTotal

    # Simulate Poisson point process
    # Poisson number of points
    numbPoints = sim_rng.poisson(lambda0 * areaTotal)
    # angular coordinates
    theta = 2 * np.pi * sim_rng.uniform(0, 1, numbPoints)
    # radial coordinates
    rho = sim_radius * np.sqrt(sim_rng.uniform(0, 1, numbPoints))

    # Convert from polar to Cartesian coordinates
    xx = rho * np.cos(theta)
    yy = rho * np.sin(theta)

    # Shift centre of disk to (xx0,yy0)
    xx = xx + xx0
    yy = yy + yy0
    return np.column_stack((xx, yy))


def hex_grid_setup(origin: tuple = (0, 0), isd: float = 500.0, sim_radius: float = 1000.0):
    """
    Creates a hex grid of 19 sites (57 sectors) using the hexalattice module when using the defaults.
    Returns the
    """
    fig, ax = plt.subplots()

    hexgrid_xy, _ = create_hex_grid(nx=5,
                                    ny=5,
                                    min_diam=isd,
                                    crop_circ=sim_radius,
                                    align_to_origin=True,
                                    edge_color=[0.75, 0.75, 0.75],
                                    h_ax=ax,
                                    do_plot=True)

    hexgrid_x = hexgrid_xy[:, 0]
    hexgrid_y = hexgrid_xy[:, 1]
    circle_dashed = plt.Circle(
        origin, sim_radius, fill=False, linestyle='--', color='r')

    ax.add_patch(circle_dashed)
    ax.scatter(hexgrid_x, hexgrid_y, marker='2')
    # Factor to set the x,y-axis limits relative to the isd value.
    ax_scaling = 3 * isd + 500
    ax.set_xlim([-ax_scaling, ax_scaling])
    ax.set_ylim([-ax_scaling, ax_scaling])
    ax.set_aspect('equal')
    return hexgrid_xy, fig


def fig_timestamp(fig, author='', fontsize=6, color='gray', alpha=0.7, rotation=0, prespace='  '):
    # Keith Briggs 2020-01-07
    # https://riptutorial.com/matplotlib/example/16030/coordinate-systems-and-text
    date = strftime('%Y-%m-%d %H:%M', localtime())
    fig.text(  # position text relative to Figure
        0.01, 0.005, prespace + '%s %s' % (author, date,),
        ha='left', va='bottom', fontsize=fontsize, color=color,
        rotation=rotation,
        transform=fig.transFigure, alpha=alpha)


def write_args_to_json(outfile, parse_args_obj):
       if not outfile.endswith('.json'):
            outfile += '.json'
            # convert namespace to dictionary
            args_dict = vars(parse_args_obj)
            # write to json file
            with open(outfile, 'w') as f:
                json.dump(args_dict, f, indent=4)


def main(seed, isd, sim_radius, power_dBm, nues, until, author=None):
    # Create a simulator object
    sim = Sim(rng_seed=seed)

    # Create the 19-cell hex-grid and place Cell instance at the centre
    sim_hexgrid_centres, hexgrid_plot = hex_grid_setup(isd=isd, sim_radius=sim_radius)
    for centre in sim_hexgrid_centres[:]:
        x, y = centre
        z = 25.0
        sim.make_cell(interval=until * 1e-2,
                      xyz=[x, y, z], power_dBm=power_dBm)

    # Generate UEs using PPP and add to simulation
    ue_ppp = generate_ppp_points(sim=sim, expected_pts=nues, sim_radius=sim_radius)
    for i in ue_ppp:
        x, y = i
        ue_xyz = x, y, 1.5
        sim.make_UE(xyz=ue_xyz).attach_to_strongest_cell_simple_pathloss_model()

    # Add the logger to the simulator
    sim.add_logger(MyLogger(sim, logging_interval=5))

    # Plot setup if desired (uncomment to activate)
    # plt.scatter(x=ue_xyz[0], y=ue_xyz[1], s=1.0)
    # fig_timestamp(fig=hexgrid_plot, author='Kishan Sthankiya')

    # Run simulator
    sim.run(until=until)


if __name__ == '__main__':  # run the main script

    # Create cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-seeds', type=int, default=0, help='seed value for random number generator')
    parser.add_argument('-isd', type=float, default=500.0, help='Base station inter-site distance in metres')
    parser.add_argument('-sim_radius', type=float, default=1000.0, help='Simulation bounds radius in metres')
    parser.add_argument('-power_dBm', type=float, default=30.0, help='Cell transmit power in dBm.')
    parser.add_argument('-nues', type=int, default=10, help='number of UEs')
    parser.add_argument('-until', type=float, default=1.0,  help='simulation time')

    # Create the args namespace
    args = parser.parse_args()

    # Write input arguments to file for reference (uncomment to activate)
    # outfile = '_'.join([script_name,'config',timestamp_iso])
    # write_args_to_json(outfile=outfile, parse_args_obj=args)

    # Run the __main__
    main(seed=args.seeds, isd=args.isd, sim_radius=args.sim_radius, power_dBm=args.power_dBm, nues=args.nues,
         until=args.until)