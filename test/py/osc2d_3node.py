import os
import time

import numpy as np
import scipy as sc
import scipy.io as sio

from tvb.simulator.lab import *
from utils.pyutils.connMatrixPlotter import connMatrixPlotter

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import product

# Set the directories for saving figures and data
figureDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/figures/osc2d_3node/'
dataDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/data/osc2d_3node/'

# Create simulation

surrogate_connectivity = connectivity.Connectivity(number_of_regions=3,
                                                   number_of_connections=9,
                                                   weights=np.array([[0, 15, 0], [1, 0, 1], [0, 15, 0]]),
                                                   tract_lengths=np.array([[0, 10, 0], [10, 0, 3], [0, 3, 0]]),
                                                   region_labels=np.array(['DLPFC', 'IPSc', 'v1']),
                                                   centres=np.array([[0, 0, 0], [1, 1, 1]])
                                                   )
connMatrixPlotter(surrogate_connectivity)


output_setting = simulator.monitors.SubSample(period=3.90625)
output_setting.configure()
simulation = simulator.Simulator(connectivity=surrogate_connectivity,
                                 coupling=coupling.Linear(),
                                 integrator=integrators.HeunStochastic(dt=0.5, noise=noise.Additive(nsig=np.array([0.01]))),
                                 model=models.Generic2dOscillator(a=np.array([2.2]),
                                                                  b=np.array([-1.0]),
                                                                  c=np.array([0.0]),
                                                                  d=np.array([0.1]),
                                                                  I=np.array([0.0]),
                                                                  alpha=np.array([1.0]),
                                                                  beta=np.array([0.2]),
                                                                  gamma=np.array([-1.0]),
                                                                  e=np.array([0.0]),
                                                                  g=np.array([1.0]),
                                                                  f=np.array([0.333]),
                                                                  tau=np.array([1.25])),
                                 monitors=[output_setting],
                                 simulation_length=5000
                                 )
simulation.configure()


def run_sim(global_coupling, noise):
    simulation.coupling.a = global_coupling
    simulation.integrator.noise.nsig = noise
    print("Starting Generic2dOscillator simulation with coupling factor " + str(global_coupling))
    results = simulation.run()
    time = results[0][0].squeeze()
    data = results[0][1].squeeze()
    return (global_coupling, time, data)

# gc_range = np.arange(0.35, 0.7, .03)
# data = []
# for gc in gc_range:
#     data.append((run_sim(np.array([gc]))))




global_coupling = np.r_[0.3:0.7:0.1]
noise = np.r_[0:0.04:0.01]
conduction_speed = np.r_[0:20:5]

for (ai, bi) in list(product(*[global_coupling, noise])):
    run_sim(np.array([ai]), np.array([bi]))





# fileNameTemplate = r'/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/figures/osc2d_3node/oscPlot_{0:02d}.png'
# for i in range(len(data)):
#     fig, ax = plt.subplots()
#     ax.set_title('3 Coupled Generic3dOscillators with Global Coupling = ' + str(data[i][0]))
#     ax.set_xlabel('Time (ms)')
#     ax.set_ylabel('Oscillatory activity')
#     ax.plot(data[i][2])
#     ax.legend(['DLPFC', 'IPSc', 'V1'], loc='upper right')
#     plt.savefig(fileNameTemplate.format(i), format='png')
#     plt.clf()
#
# for i in range(len(data)):
#     if os.path.exists(dataDir + 'osc2d_3node_0.mat'):
#         sio.savemat(dataDir + 'osc2d_3node_{0}_{1}.mat'.format(i, int(time.time())), {'data': data[i][2]})
#     else:
#         sio.savemat(dataDir + 'osc2d_3node_0.mat', {'data': data[i][2]})

