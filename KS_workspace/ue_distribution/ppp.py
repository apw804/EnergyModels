# 2022-11-03 21:20:58
# Kishan Sthankiya
# Module to simulate Poisson Point Process for a single point in a defined 2D plane.
import matplotlib.pyplot as plt
import numpy as np

seed = 0
PPP_rng = np.random.default_rng(0)


# Poisson Point Process: UE position as xy
def ppp(nues, x_max, y_max, x_min=0, y_min=0):
    n_points = PPP_rng.poisson(nues)  
    x = (x_max*np.random.uniform(0, 1,n_points)+x_min)
    y = (y_max*np.random.uniform(0, 1,n_points)+y_min)
    return np.stack((x, y), axis=1)

n_ues = 1e3
t_x_max = 1000
t_y_max = 1000
a = ppp(n_ues,t_x_max,t_y_max)

print(a)
print(type(a))

# Plot this point UE (for illustration only)
plt.scatter(x=a[:,0], y=a[:,1], edgecolor='b', facecolor='none', alpha=0.5)
plt.xlim(0, t_x_max)
plt.ylim(0, t_y_max)
plt.xlabel('x')
plt.ylabel('y')
plt.plot()
plt.show()
