# Kishan Sthankiya

import numpy as np
from math import pi

def generate_PPP_points_radii(lamb,npts=1000):
    '''Generates npts number of UEs, distributed according to a homogeneous PPP
    with intensity lamb and returns an array of distances to the origin.'''
    pi_lamb=pi*lamb
    return np.sqrt(np.cumsum(np.random.uniform(1.0,npts))/pi_lamb)

print(generate_PPP_points_radii())