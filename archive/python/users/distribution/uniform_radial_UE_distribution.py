import numpy as np
from matplotlib import pyplot as plt
from numpy import pi


def generate_ppp_points(n_pts=100, sim_radius=500.0):
    """
    Generates npts number of points, distributed according to a homogeneous PPP
    with intensity lamb and returns an array of distances to the origin.
    """

    n = 0
    xx = []
    yy = []
    while n < n_pts:
        # Generate the radius value
        radius_polar = sim_radius * np.sqrt(np.random.uniform(0, 1, 1))

        # Generate theta value
        theta = np.random.uniform(0, 2 * pi, 1)

        # Convert to cartesian coords
        x = radius_polar * np.cos(theta)
        y = radius_polar * np.sin(theta)

        # Add to the array
        xx.append(float(x))
        yy.append(float(y))

        n = n + 1

    if n == n_pts:
        return np.array(list(zip(xx, yy)))


b = generate_ppp_points()
print(b)

plt.scatter(b[:, 0], b[:, 1], marker=".")
plt.axis('equal')
plt.show()
