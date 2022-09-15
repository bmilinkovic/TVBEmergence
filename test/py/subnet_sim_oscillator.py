import os

import tvb.simulator.plot.plotter
from tvb.simulator.lab import *
import numpy as np
import matplotlib.pyplot as plt
from utils.pyutils.connMatrixPlotter import connMatrixPlotter
from networks.pynetworks.subset9Modular36 import subnet9mod36
import time
import scipy.io as sio

import seaborn as sns


# Setting up saving options

figureDir = "/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/figures/"
savefig = True

dataDir = "/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/data/"
resultsmat = True

# Starting analysis

conn = subnet9mod36()
monitors = simulator.monitors.SubSample(period=.9765625) # subsampling at 256Hz or every 3.90625 ms
monitors.configure()



# Set up the simulation with all the correct arguments.
simulation = simulator.Simulator(connectivity=connectivity.Connectivity.from_file(),
                                 coupling=coupling.Linear(),
                                 integrator=integrators.HeunStochastic(dt=0.5, noise=noise.Additive(nsig = np.array([0.01]))),
                                 model=models.Generic2dOscillator(a=np.array([2.2]), b=np.array([-1.0]), c=np.array([0.0]),
                                                                  d=np.array([0.1]), I=np.array([0.0]), alpha=np.array([1.0]),
                                                                  beta=np.array([0.2]), gamma=np.array([-1.0]), e=np.array([0.0]),
                                                                  g=np.array([1.0]), f=np.array([0.333]), tau=np.array([1.25])),
                                 monitors=[monitors],
                                 simulation_length=1000)
simulation.configure()


# Defining a function to run a simulation
def run_sim(global_coupling):
    simulation.coupling.a = global_coupling
    print("Starting Generic2dOscillator simulation with coupling factor " + str(global_coupling))
    results = simulation.run()
    time_line = results[0][0].squeeze()
    data = results[0][1].squeeze()
    return (global_coupling, time_line, data)

# running a parameter sweep of the global coupling parameter
gc_range = np.arange(0.35, 0.7, .03)
data = []
for gc in gc_range:
    data.append((run_sim(np.array([gc]))))

# computing correlation coefficient

def compute_corr(time_line, data_result, sim):
    input_shape = data_result.shape
    result_shape = (input_shape[1], input_shape[1])
    sample_period = sim.monitors[0].period
    t_start = sample_period
    t_end = time_line[-1]
    t_lo = int((1. / sample_period) * (t_start - sample_period))
    t_hi = int((1. / sample_period) * (t_end - sample_period))
    t_lo = max(t_lo,0)
    t_hi = max(t_hi, input_shape[0])
    FC = np.zeros(result_shape)
    FC[:,:] = np.corrcoef(data_result.T)
    return FC

uidx = np.tril_indices(76,1)
sim_FC = np.arctanh(FC)[uidx]


time_line = data[0][1]
for i in range(len(data)):
    FC = compute_corr(time_line, data[i][2], simulation)
    #plot functional connectivity
    mask = np.triu(np.ones_like(FC, dtype=bool))
    f, ax = plt.subplots(figsize=(11,9))
    cmap = sns.diverging_palette(230,20, as_cmap=True)
    sns.heatmap(FC, mask=mask, cmap=cmap, vmax=FC.max(), vmin=FC.min(), center=0, square=True, linewidths=.5,
                cbar_kws={"shrink":.5}, xticklabels=simulation.connectivity.region_labels, yticklabels=simulation.connectivity.region_labels)
    plt.show()



plt.figure(1)
connMatrixPlotter.connMatrixPlotter(conn)
plt.show()


# UNDER CONSTRUCTION - PLOTTING TIMESERIES PROPERLY
# MATPLOTLIB example that doesn't currently work.

# nodes, time = data[0][1]
# segs= []
# for i in range(nodes):
#     segs.append(np.hstack((data[0][1][:,i]))
#
# offsets = np.zeros((nodes, 2), dtype=float)
#
# line_segments = matplotlib.collections.LineCollections(segs, offsets=offsets)






# plotting time-series of oscillatory activity and saving

# if savefig:
#     fileNameTemplate = r'/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/figures/oscPlot_{0:02d}.png'
#
#     for i in range(len(data)):
#         plt.figure()
#         plt.xlabel('Time (ms)')
#         plt.ylabel('Oscillatory activity')
#         plt.plot(data[i][1])
#         plt.savefig(fileNameTemplate.format(i), format='png')
#         plt.clf()
#
# # SAVING DATA FOR MVGC USE
# if resultsmat:
#     for i in range(len(data)):
#         if os.path.exists(dataDir + 'oscSim.mat'):
#             sio.savemat(dataDir + 'oscSim_{0}_{1}.mat'.format(i, int(time.time())), {'data': data[i][1]})


