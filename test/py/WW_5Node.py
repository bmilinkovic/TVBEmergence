import numpy as np
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

from tvb.simulator.lab import *

# %%
'''
Preparation of results directory
'''
resultsDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/data'
figureDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/figures'

# connectivity
default = connectivity.Connectivity.from_file()
default.configure()

# get a row and columnar index for specific regions or nodes
idx = np.r_[random.sample(range(0, 75), 5)]

# configure weights structural and connectivity
changedWeights = np.array([[9, 9, 9, 0, 0],
                           [9, 9, 9, 0, 0],
                           [9, 9, 9, 0, 0],
                           [0, 0, 0, 9, 9],
                           [0, 0, 0, 9, 9]])

tracts_nodelay = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])

subset_nodelay = connectivity.Connectivity(weights=changedWeights,
                                           tract_lengths=default.tract_lengths[idx][:, idx],
                                           centres=default.centres[idx],
                                           region_labels=default.region_labels[idx])
subset_nodelay.configure()

# %%

# configure monitors
monitors = monitors.SubSample(period=1.953125)

# configure simulation
simulation = simulator.Simulator(connectivity=subset_nodelay,
                                 coupling=coupling.Linear(),
                                 # remember that the integration stepsize needs to use base 2 for numeric stability.
                                 # also, it seems I had problem running anything larger than 2**-6 as a step size, and
                                 # numerical instability occurred.
                                 integrator=integrators.HeunStochastic(dt=2 ** -6,
                                                                       noise=noise.Additive(nsig=np.array([0.671]))),
                                 monitors=[monitors],
                                 model=models.ReducedWongWangExcInh(),
                                 simulation_length=5000)
simulation.configure()


