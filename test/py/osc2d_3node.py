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
figureDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/osc2d_nodelay_3node/figures'
dataDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/osc2d_nodelay_3node/data/'

# Create simulation

surrogate_connectivity = connectivity.Connectivity(number_of_regions=3,
                                                   number_of_connections=9,
                                                   weights=np.array([[0, 1, 0],
                                                                     [1, 0, 0],
                                                                     [0, 0, 0]]),
                                                   tract_lengths=np.array([[0, 0, 0],
                                                                           [0, 0, 0],
                                                                           [0, 0, 0]]),
                                                   region_labels=np.array(['DLPFC', 'IPSc', 'v1']),
                                                   centres=np.array([[0, 0, 0], [1, 1, 1]])
                                                   )
connMatrixPlotter(surrogate_connectivity)

output_setting = simulator.monitors.SubSample(period=3.90625)
output_setting.configure()
simulation = simulator.Simulator(connectivity=surrogate_connectivity,
                                 coupling=coupling.Linear(),
                                 integrator=integrators.HeunStochastic(dt=0.5, noise=noise.Additive()),
                                 model=models.Generic2dOscillator(a=np.array([2.2]),
                                                                  b=np.array([-1.0]),
                                                                  c=np.array([0.0]),
                                                                  d=np.array([0.1]),
                                                                  I=np.array([0.0]),
                                                                  alpha=np.array([1.0]),
                                                                  beta=np.array([0.2]),
                                                                  gamma=np.array([1.0]),
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
    #simulation.conduction_speed = conduction_speed
    print("Starting Generic2dOscillator simulation with coupling factor: " + str(global_coupling) + " noise: " + str(noise))
    results = simulation.run()
    time = results[0][0].squeeze()
    data = results[0][1].squeeze()
    return (global_coupling, noise, time, data)

# setting the sweep

global_coupling_log = 10**np.r_[-5:0.7:20j]
noise_log = 10**np.r_[-6:-1:20j]
#conduction_speed = np.r_[0:21:5]

# beginning the simulation
data = []
for (ai, bi) in list(product(*[global_coupling_log, noise_log])):
    data.append(run_sim(np.array([ai]), np.array([bi])))

#Save the data

for i in range(len(data)):
        sio.savemat(dataDir + 'osc2d_nodelay_3node_gc-{0:02f}_noise-{1:02f}.mat'.format(float(data[i][0]), float(data[i][1])), {'data': data[i][3]})



fileNameTemplate = r'/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/osc2d_nodelay_3node/figures/oscPlot_{0:02d}.svg'
for i in range(len(data)):
    fig, ax = plt.subplots()
    ax.set_title('3 Coupled Generic3dOs:cillators with GC = {0}, Noise = {1}'.format(str(data[i][0]), str(data[i][1])), fontsize=10, fontweight='bold')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Oscillatory activity')
    ax.plot(data[0][2], data[i][3] + np.r_[:len(surrogate_connectivity.weights)], linewidth=0.4)
    ax.set_yticks(np.arange(len(surrogate_connectivity.region_labels)), surrogate_connectivity.region_labels, fontsize=10)
    right_side = ax.spines["right"]
    top_side = ax.spines["top"]
    right_side.set_visible(False)
    top_side.set_visible(False)
    plt.savefig(fileNameTemplate.format(i), format='svg')
    plt.clf()


