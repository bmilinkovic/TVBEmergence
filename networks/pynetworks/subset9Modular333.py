from tvb.simulator.lab import *
import numpy as np
import utils.connMatrixPlotter

# configure the connectivity object _default_ as the default connectivity from data files
default = connectivity.Connectivity.from_file()
default.configure()

# get a row and columnar index for specific regions or nodes
# idx = np.r_[41, 45, 49, 52, 56, 59, 63, 67, 73] # to keep tract lengths the same across simulations choose this option

idx = np.r_[random.sample(range(0,75),9)]

# set a weights array for those indexes which will only change the weights with the tract matrix remaining the
# same as the tract matrix of the default connectome
changedWeights = np.array([[0, 9, 9, 0, 0, 0, 0, 0, 0],
                           [9, 0, 9, 0, 0, 0, 0, 0, 0],
                           [9, 9, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 9, 9, 0, 0, 0],
                           [0, 0, 0, 9, 0, 9, 0, 0, 0],
                           [0, 0, 0, 9, 9, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 9, 9],
                           [0, 0, 0, 0, 0, 0, 9, 0, 9],
                           [0, 0, 0, 0, 0, 0, 9, 9, 0]])

subset = connectivity.Connectivity(weights=changedWeights,
                                   tract_lengths=default.tract_lengths[idx][:, idx],
                                   centres=default.centres[idx],
                                   region_labels=default.region_labels[idx])
subset.configure()

utils.connMatrixPlotter.connMatrixPlotter(subset)