import numpy as np
import random
from tvb.simulator.lab import *
from scipy.stats import zscore
import scipy.io as sio
import pandas as pd
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

import utils.pyutils.connMatrixPlotter
# from src.networks.subset9Modular36 import subnet9mod36

# %%
'''
Preparation of results directory
'''
resultsDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/data'
figureDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/figures'

# %%
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
                                 model=models.ReducedSetHindmarshRose(),
                                 simulation_length=5000)
simulation.configure()

# run_sim will run the simulation over a particular coupling range. Within the function it will also prepare the
# simulated data by extract the _xi_ variable that describes the dynamics of the local field potential based on the
# excitatory population and will sum over the modes and then z-score each of the local dynamics independently before
# constructing a structure with the entire time-series called _data_cleaned_. The data then needs to be transposed to
# be plotted properly by matplotlib.
def run_sim(global_coupling):
    simulation.coupling.a = global_coupling
    print("Starting SJ3D simulation with coupling factor " + str(global_coupling))
    results = simulation.run()
    data = results[0][1].squeeze()
    data_cleaned = np.zeros([5, 2060])  # initialise structure for z-scored data.
    for i in range(len(changedWeights)):
        data_cleaned[i] = zscore(np.sum(data[500:, 0, i, :], axis=1))
    return data_cleaned.T

# defining a multiparameter simulation run.
# def run_sim_multiparameter(gc, tl, noise):
#     connectivity =
#     simulation = simulator.Simulator(connectivity = connectivity,
#                                      coupling = coupling.Linear(a=np.array([gc])),
#                                      integrator = integrators.HeunStochastic(dt = 2**-6, noise.Additive(nsig=np.array([noise]))),
#                                      monitors = [monitors.SubSample(period=1.953125)],
#                                      model = models.ReducedSetHindmarshRose(),
#                                      simulation_length = 5000)
#     simulation.configure()
#     print("Starting a multiparameter sweet of a SJ3D simulation with coupling factor: " + str(gc) + " and noise: " + str(noise))
#     results = simulation.run()
#     data = results[0][1].squeeze()
#     data.cleaned = np.zeros([5, 2060])
#     for i in range(len(changedWeights)):
#         data_cleaned[i] = zscore(np.sum(data[500:,0,i,:], axis=1))
#     return data_cleaned.T
# %%

gc_range = np.arange(0.0, .2, .05)
# the below could be used for a parameter sweep across three different paramers. these are lists to iterate over for
# tract lengths as well as noise. However, for this a new run_sim function needs to be created to accomodated for
# multiple parameter sweeps.

# tl_range = np.arange(0.0, 50, 5)
# noise_range = np.arange(0.0, 1, 0.1)

# initialising cleaned data structure and then running the simulation _run_sim_ over the global coupling range.
data_cleaned = []
for gc in gc_range:
    data_cleaned.append(run_sim(np.array([gc])))

# %%
# plot data
fig, axs = plt.subplots(4)
fig.suptitle("5-node network across Global Coupling range")
axs[0].plot(data_cleaned[0])
axs[1].plot(data_cleaned[1])
axs[2].plot(data_cleaned[2])
axs[3].plot(data_cleaned[3])
plt.savefig(os.path.join(figureDir,'SJ3D-node-5-paramsweep-gc.svg'))
plt.show()


# %%
# saving data file.

sio.savemat(os.path.join(resultsDir, 'SJ3D-node-5-paramsweep-gc.mat'), {"data": data_cleaned})



