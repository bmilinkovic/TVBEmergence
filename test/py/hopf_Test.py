import time as tm
import os
from itertools import product

import numpy as np
import pandas as pd
import scipy as sc
import scipy.io as sio

import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import tvb.simulator.models.oscillator

from tvb.simulator.lab import *

from utils.pyutils.connMatrixPlotter import connMatrixPlotter

#%% Setting up a connectivity.

surrogate_connectivity = connectivity.Connectivity(number_of_regions=3,
                                                   number_of_connections=9,
                                                   weights=np.array([[0, 1, 0],
                                                                     [1, 0, 0],
                                                                     [0, 0, 0]]),
                                                   tract_lengths=np.array([[0, 10, 0],
                                                                           [10, 0, 3],
                                                                           [0, 3, 0]]),
                                                   region_labels=np.array(['DLPFC', 'IPSc', 'v1']),
                                                   centres=np.array([[0, 0, 0], [1, 1, 1]])
                                                   )
connMatrixPlotter(surrogate_connectivity)
plt.show()


output_setting = simulator.monitors.SubSample(period=3.90625)
output_setting.configure()
simulation = simulator.Simulator(connectivity=surrogate_connectivity,
                                 coupling=coupling.Linear(),
                                 integrator=integrators.HeunStochastic(dt=0.5, noise=noise.Additive()),
                                 model=tvb.simulator.models.oscillator.SupHopf(a=np.array([-0.01])),
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
    data_cleaned = results[0][1].squeeze()
    return (global_coupling, noise, data_cleaned, time)

global_coupling_log = 10**np.r_[-5:0:4j]
noise_log = 10**np.r_[-6:-2:4j]

global_coupling = [0, 0.3, 0.6, 0.9]
noise = [0, 0.01, 0.02, 0.03]

data = []
for (ai, bi) in list(product(*[global_coupling, noise])):
    data.append(run_sim(np.array([ai]), np.array([bi])))

    for i in range(len(data)):
        fig, ax = plt.subplots()
        ax.set_title('3 Coupled supHopf with GC = {0}, Noise = {1}'.format(str(data[i][0]), str(data[i][1])),
                     fontsize=10, fontname='Times New Roman', fontweight='bold')
        ax.set_xlabel('Time (ms)', fontsize=8, fontname='Times New Roman', fontweight='bold')
        ax.set_ylabel('Oscillatory activity')
        right_side = ax.spines["right"]
        top_side = ax.spines["top"]
        ax.plot(data[i][2].T, linewidth=0.4)
        ax.legend(['DLPFC', 'IPSc', 'V1'], loc='upper right')
        ax.axvspan(0, 500, alpha=0.5, color='grey')
        plt.show()


