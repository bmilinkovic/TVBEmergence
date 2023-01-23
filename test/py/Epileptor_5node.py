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
import time as tm
from utils.pyutils.connMatrixPlotter import connMatrixPlotter

#%% initialise directories for results

resultsDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/'
dataDir = os.path.join(resultsDir, 'Epileptor_5node/data/')
figureDir = os.path.join(resultsDir, 'Epileptor_5node/figures/')
connDir = os.path.join(resultsDir, 'Epileptor_5node/conn/')

if not os.path.exists(figureDir):
   os.makedirs(figureDir)

if not os.path.exists(dataDir):
    os.makedirs(dataDir)

if not os.path.exists(connDir):
    os.makedirs(connDir)


#%% initialise connectivity



default = connectivity.Connectivity.from_file()
number_of_regions = default.weights.shape[0]
default.weights - default.weights * np.eye(number_of_regions, number_of_regions)    # subtracting self connections
default.weights = default.weights / np.abs(default.weights.max())                   # normalising the weights
default.configure()

idx = np.r_[[40, 47, 53, 58, 62, 69, 72, 75]]

subset = connectivity.Connectivity(weights=default.weights[idx][:, idx],
                                           tract_lengths=default.tract_lengths[idx][:, idx],
                                           centres=default.centres[idx],
                                           region_labels=default.region_labels[idx])
subset.configure()


connMatrixPlotter(default)     # plot connectivity
plt.savefig(os.path.join(connDir, 'Epileptor_5node_connectivity_full.svg'), format='svg')
plt.show()

connMatrixPlotter(subset)
plt.savefig(os.path.join(connDir, 'Epileptor_5node_connectivity_subset.svg'), format='svg')
plt.show()

#%% initialise complete local model

Epileptor = models.Epileptor()
# Ks=np.array([1]),r=np.array([0.00015]) this goes into the epliptor if its 2D
Epileptor.x0 = np.ones((8))*-2.4                                 # sets the Healthy Zone (HZ) dynamics
Epileptor.x0[[0, 1, 4]] = np.ones((3))*-1.6                      # sets the Epileptogenic Zone (EZ) dynamics
Epileptor.x0[[5, 6]] = np.ones((2))*-1.8                         # sets the Propagation Zone (PZ) dynamics

#%% initialise conditions

Epileptor.state_variable_range["x1"] = np.array([-1.8, -1.8])
Epileptor.state_variable_range["z"] = np.array([3.6, 3.6])

#%% initialise simulator

sim = simulator.Simulator(model=Epileptor,
                          connectivity=subset,
                          coupling=coupling.Difference(a=np.array([-0.2])),
                          integrator=integrators.HeunDeterministic(dt=0.5),
                          monitors=[monitors.TemporalAverage(period=3.90625)])
sim.configure()


#%% run simulation

tic = tm.time()

# Single Simulation
# tavg_time, tavg_data = [], []
# for tavg in sim(simulation_length=5000):
#     if not tavg is None:
#         tavg_time.append(tavg[0][0])
#         tavg_data.append(tavg[0][1])



def run_sim(global_coupling, noise):
    simulation.coupling.a = global_coupling
    simulation.integrator.noise.nsig = noise
    print("Starting Epileptor simulation with coupling factor: " + str(global_coupling) + " and noise: " + str(noise))
    results = simulation.run()
    print( "Finished simulation! Executed in " + str(tm.time()-tic))
    time = results[0][0].squeeze()
    data = results[0][1].squeeze()
    data_cleaned = np.zeros([5, 1024])  # initialise structure for z-scored data.
    for i in range(len(changedWeights)):
        data_cleaned[i] = zscore(np.sum(data[:, 0, i, :], axis=1))
    return (global_coupling, noise, data_cleaned, time)

print('simulation required %0.3f seconds.' % (tm.time() - tic))

# global_coupling_log = 10**np.r_[-2:-0.5:15j]
# noise_log = 10**np.r_[-3:-0.002:15j]
#
# data = []
# for (ai, bi) in list(product(*[global_coupling_log, noise_log])):
#     data.append(run_sim(np.array([ai]), np.array([bi])))
#

# #%% Save Data
#
# for i in range(len(data)):
#         sio.savemat(dataDir + 'Epileptor_5node_gc-{0:02f}_noise-{1:02f}.mat'.format(float(data[i][0]), float(data[i][1])), {'data': data[i][2]})
#
#%% Normalise time-series for nice plots
tavg_data /= (np.max(tavg_data, 0) - np.min(tavg_data, 0))
tavg_data -= np.mean(tavg_data, 0)

#%% Plot the figures
fig, ax = plt.subplots()
ax.set_title("8-node Epileptor time series", fontsize=24)
ax.plot(tavg_time[:], tavg_data[:, 0, :, 0] + np.r_[:8], linewidth=0.4)
ax.set_xlabel('Time (ms)', fontsize=18, fontweight='bold')
ax.set_yticks(np.arange(len(sim.connectivity.region_labels)), sim.connectivity.region_labels, fontsize=10)
right_side = ax.spines["right"]
top_side = ax.spines["top"]
right_side.set_visible(False)
top_side.set_visible(False)
ax.axvspan(0, 500, alpha=0.5, color='grey')
plt.savefig(os.path.join(connDir, 'Epileptor_5node_noise'), format='svg')
plt.show()



#
# fileNameTemplate = r'/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/Epileptor_5node/figures/Epileptor_5node_gc-{0:02f}_noise-{1:02f}.svg'
# for i in range(len(data)):
#     fig, ax = plt.subplots()
#     ax.set_title('5 coupled Epileptor Models with GC={0:02f} and Noise={1:02f}'.format(float(data[i][0]), float(data[i][1])), fontsize=10, fontname='Times New Roman', fontweight='bold')
#     ax.set_xlabel('Time (ms)', fontsize=8, fontname='Times New Roman', fontweight='bold')
#     ax.set_ylabel('Local Field Potential (LFP)', fontsize=8, fontname='Times New Roman', fontweight='bold')
#     ax.tick_params(axis='both', which='major', labelsize=7)
#     right_side = ax.spines["right"]
#     top_side = ax.spines["top"]
#     right_side.set_visible(False)
#     top_side.set_visible(False)
#     ax.plot(data[0][3], data[i][2].T, linewidth=0.4)  # hacked the time because time is cumulative in the plots
#     ax.axvspan(0, 500, alpha=0.5, color='grey')
#     ax.legend(['Node[1]', 'Node[2]', 'Node[3]', 'Node[4]', 'Node[5]'], loc='upper right', fontsize=6)
#     plt.savefig(fileNameTemplate.format(float(data[i][0]), float(data[i][1])), format='svg')
#     plt.clf()
#
