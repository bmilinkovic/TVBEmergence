import numpy as np
import random
from tvb.simulator.lab import *
from scipy.stats import zscore
import scipy.io as sio
import pandas as pd
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from itertools import product
# this below code needs to be done if the primary script you are
# running is in a different directory to the subpackage or module
# this is slightly annoying, and I need to figure a better work around
# if at all possible.

# import sys
# sys.path.append('/Users/borjan/code/python/TVBEmergence/')
from utils.pyutils.connMatrixPlotter import connMatrixPlotter



# %%
'''
Preparation of results directory
'''

# Set the directories for saving figures and data
resultsDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/'
dataDir = os.path.join(resultsDir, 'SJ3D_NOCONN_5node_withlink_ps_gc-noise/data/')
figureDir = os.path.join(resultsDir, 'SJ3D_NOCONN_5node_withlink_ps_gc-noise/figures/')
connDir = os.path.join(resultsDir, 'SJ3D_NOCONN_5node_withlink_ps_gc-noise/conn/')

if not os.path.exists(figureDir):
   os.makedirs(figureDir)

if not os.path.exists(dataDir):
    os.makedirs(dataDir)

if not os.path.exists(connDir):
    os.makedirs(connDir)


# %%
# connectivity
default = connectivity.Connectivity.from_file()
default.configure()

# get a row and columnar index for specific regions or nodes
idx = np.r_[random.sample(range(0, 75), 5)]

# configure weights structural and connectivity
changedWeights = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])

tracts_nodelay = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])

subset = connectivity.Connectivity(weights=changedWeights,
                                           tract_lengths=default.tract_lengths[idx][:, idx],
                                           centres=default.centres[idx],
                                           region_labels=default.region_labels[idx])
subset.configure()


connMatrixPlotter(subset)
plt.savefig('/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/SJ3D_NOCONN_5node_withlink_ps_gc-noise/conn/SJ3D_NOCONNN_5node_withlink_gc.svg', format='svg')
plt.show()


# %% CONFIGURE THE SIMULATION

# configure monitors
monitors = monitors.TemporalAverage(period=3.90625)

# configure simulation
simulation = simulator.Simulator(connectivity=subset,
                                 coupling=coupling.Linear(),
                                 integrator=integrators.HeunStochastic(dt=2**-6,
                                                                       noise=noise.Additive()),
                                 monitors=[monitors],
                                 model=models.ReducedSetHindmarshRose(),
                                 simulation_length=4000)
simulation.configure()

# run_sim will run the simulation over a particular coupling range. Within the function it will also prepare the
# simulated data by extract the _xi_ variable that describes the dynamics of the local field potential based on the
# excitatory population and will sum over the modes and then z-score each of the local dynamics independently before
# constructing a structure with the entire time-series called _data_cleaned_. The data then needs to be transposed to
# be plotted properly by matplotlib.
def run_sim(global_coupling, noise):
    simulation.coupling.a = global_coupling
    simulation.integrator.noise.nsig = noise
    print("Starting SJ3D simulation with coupling factor: " + str(global_coupling) + " and noise: " + str(noise))
    results = simulation.run()
    time = results[0][0].squeeze()
    data = results[0][1].squeeze()
    data_cleaned = np.zeros([5, 1024])  # initialise structure for z-scored data.
    for i in range(len(changedWeights)):
        data_cleaned[i] = zscore(np.sum(data[:, 0, i, :], axis=1))
    return (global_coupling, noise, data_cleaned, time)


global_coupling_log = 10**np.r_[-2:-0.5:20j]
noise_log = 10**np.r_[-3:-0.002:20j]

data = []
for (ai, bi) in list(product(*[global_coupling_log, noise_log])):
    data.append(run_sim(np.array([ai]), np.array([bi])))


# %% Save Data

for i in range(len(data)):
        sio.savemat(dataDir + 'SJ3D_NOCONN_5node_withlink_gc-{0:02f}_noise-{1:02f}.mat'.format(float(data[i][0]), float(data[i][1])), {'data': data[i][2]})


fileNameTemplate = r'/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/SJ3D_NOCONN_5node_withlink_ps_gc-noise/figures/SJ3D_NOCONN_5node_withlink_gc-{0:02f}_noise-{1:02f}.svg'
for i in range(len(data)):
    fig, ax = plt.subplots()
    ax.set_title('5 SJ3D Models with GC={0:02f} and Noise={1:02f}'.format(float(data[i][0]), float(data[i][1])), fontsize=10, fontname='Times New Roman', fontweight='bold')
    ax.set_xlabel('Time (ms)', fontsize=8, fontname='Times New Roman', fontweight='bold')
    ax.set_ylabel('Local Field Potential (LFP)', fontsize=8, fontname='Times New Roman', fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=7)
    right_side = ax.spines["right"]
    top_side = ax.spines["top"]
    right_side.set_visible(False)
    top_side.set_visible(False)
    ax.plot(data[0][3], data[i][2].T, linewidth=0.4)  # hacked the time because time is cumulative in the plots
    ax.axvspan(0, 500, alpha=0.5, color='grey')
    ax.legend(['Node[1]', 'Node[2]', 'Node[3]', 'Node[4]', 'Node[5]'], loc='upper right', fontsize=6)
    plt.savefig(fileNameTemplate.format(float(data[i][0]), float(data[i][1])), format='svg')
    plt.clf()





