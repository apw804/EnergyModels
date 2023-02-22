# Single cell test / scratchpad

from AIMM_simulator import Sim, Logger, np_array_to_str, to_dB, NR_5G_standard_functions, Scenario, MME, UMa_pathloss
from hexalattice.hexalattice import *
import matplotlib.pyplot as plt


def hex_grid_setup(origin: tuple = (0, 0), isd: float = 500, sim_radius: float = 500.0):
    """
    Creates a site using the hexalattice module.
    """
    fig, ax = plt.subplots()

    hexgrid_xy, _ = create_hex_grid(nx=1,
                                    ny=1,
                                    min_diam=isd,
                                    crop_circ=sim_radius,
                                    align_to_origin=False,
                                    edge_color=[0.75, 0.75, 0.75],
                                    h_ax=ax,
                                    do_plot=True)

    hexgrid_x = hexgrid_xy[:, 0]
    hexgrid_y = hexgrid_xy[:, 1]

    ax.scatter(hexgrid_x, hexgrid_y, marker='2')
    # Factor to set the x,y-axis limits relative to the isd value.
    ax_scaling = sim_radius-250
    ax.set_xlim([-ax_scaling, ax_scaling])
    ax.set_ylim([-ax_scaling, ax_scaling])
    ax.set_aspect('equal')
    return hexgrid_xy, fig


def main():
  sim=Sim(rng_seed=0, params={'h_UT':1.5, 'h_BS':25.0})

  # Create the hex-grid and place Cell instance at the centre
  sim_hexgrid_centres, hexgrid_plot = hex_grid_setup(isd=1000, sim_radius=1000)
  for centre in sim_hexgrid_centres[:]:
      x, y = centre
      z = 25.0
      # Create the cell
      sim.make_cell(xyz=[x, y, z], power_dBm=30.0, h_BS=25.0)
  cell0 = sim.cells[0]
  
  # Create instance of UMa-NLOS pathloss model
  pl_uma_nlos = UMa_pathloss(LOS=False, h_UT=1.5)

  # Create UE[1] at Cell Edge (500.,0.,1.5)
  ue1_xyz = [-500.,0.,1.5]
  sim.make_UE(xyz=ue1_xyz, pathloss_model=pl_uma_nlos).attach_to_strongest_cell_simple_pathloss_model()
  ue1 = sim.UEs[0]
  ue1_tp = ue1.send_subband_cqi_report()

  # Plot the UE on the axes
  plt.scatter(ue1_xyz[0],ue1_xyz[1])

  # Print sim paramaters
  print(f'\n')
  
  print(f'Base station transmit power: {cell0.power_dBm}dBm')
  print(f'Cell bandwidth: {cell0.bw_MHz}MHz')
  print(f'\n')
  print(f'UE[1] distance to cell (m): {np.linalg.norm(cell0.xyz[:1] - ue1.xyz[:1]):,.2f}')
  print(f'UE[1] UMa NLOS pathloss (dB): {pl_uma_nlos.__call__(xyz_cell=cell0.xyz, xyz_UE=ue1.xyz)}')
  print(f'UE[1] SINR: {ue1.sinr_dB[0]} (dB)')

  plt.show()

if __name__ == '__main__':
  main()
