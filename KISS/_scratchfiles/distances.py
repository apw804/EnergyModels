import numpy as np

# Generate sample data
ccp = np.random.rand(19, 2)
points = np.random.rand(401, 2)

# Calculate distance between each point in `points` and each point in `ccp`
distances = np.linalg.norm(ccp[np.newaxis, :, :] - points[:, np.newaxis, :], axis=-1)

# `distances` is now a 2D array where `distances[i, j]` is the distance between
# the i-th row in `points` and the j-th row in `ccp`

print(distances)