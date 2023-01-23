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
figureDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/figures/Hopf_3node/'
dataDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/data/Hopf_3node/'


if not os.path.exists(figureDir):
   os.makedirs(figureDir)

if not os.path.exists(dataDir):
    os.makedirs(dataDir)

# Create simulation

surrogate_connectivity = connectivity.Connectivity(number_of_regions=3,
                                                   number_of_connections=9,
                                                   weights=np.array([[2, 4, 0],
                                                                     [4, 2, 0],
                                                                     [0, 0, 1]]),
                                                   tract_lengths=np.array([[0, 10, 0],
                                                                           [10, 0, 3],
                                                                           [0, 3, 0]]),
                                                   region_labels=np.array(['DLPFC', 'IPSc', 'v1']),
                                                   centres=np.array([[0, 0, 0], [1, 1, 1]])
                                                   )
connMatrixPlotter(surrogate_connectivity)


output_setting = simulator.monitors.SubSample(period=3.90625)
output_setting.configure()
simulation = simulator.Simulator(connectivity=surrogate_connectivity,
                                 coupling=coupling.Linear(),
                                 integrator=integrators.HeunStochastic(dt=2**-3, noise=noise.Additive()),
                                 model=models.oscillator.SupHopf(a=np.array([0.0])),
                                 monitors=[output_setting],
                                 simulation_length=5000
                                 )
simulation.configure()


def run_sim(global_coupling, noise):
    simulation.coupling.a = global_coupling
    simulation.integrator.noise.nsig = noise
    #simulation.conduction_speed = conduction_speed
    print("Starting supHopf simulation with coupling factor: " + str(global_coupling) + " noise: " + str(noise))
    results = simulation.run()
    time = results[0][0].squeeze()
    data = results[0][1].squeeze()
    return (global_coupling, noise, data_cleaned, time)

#global_coupling = np.r_[0.3:0.7:0.1]
global_coupling_log = 10**np.r_[-5:0:20j]
#noise = np.r_[0:0.04:0.01]
noise_log = 10**np.r_[-6:-2:20j]
#conduction_speed = np.r_[0:21:5]

data = []
for (ai, bi) in list(product(*[global_coupling_log, noise_log])):
    data.append(run_sim(np.array([ai]), np.array([bi])))


for i in range(len(data)):
        sio.savemat(dataDir + 'Hopf_3node_gc-{0:02f}_noise-{1:02f}.mat'.format(float(data[i][0]), float(data[i][1])), {'data': data[i][2]})

fileNameTemplate = r'/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/figures/Hopf_3node/hopfPlot_gc-{0:02f}_noise-{1:02f}.svg'
for i in range(len(data)):
    fig, ax = plt.subplots()
    ax.set_title('3 Coupled supHopf with GC = {0}, Noise = {1}'.format(str(data[i][0]), str(data[i][1])), fontsize=10, fontname='Times New Roman', fontweight='bold')
    ax.set_xlabel('Time (ms)', fontsize=8, fontname='Times New Roman', fontweight='bold')
    ax.set_ylabel('Oscillatory activity')
    right_side = ax.spines["right"]
    top_side = ax.spines["top"]
    ax.plot(data[0][3], data[i][2].T, linewidth=0.4)
    ax.legend(['DLPFC', 'IPSc', 'V1'], loc='upper right')
    plt.savefig(fileNameTemplate.format(float(data[i][0]), float(data[i][1])), format='svg')
    plt.clf()


