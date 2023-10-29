import os
import time
from itertools import product
from multiprocessing import Pool, cpu_count, freeze_support

import numpy as np
import scipy as sc
import scipy.io as sio

from tvb.simulator.lab import *
from plots.ssdigraphs import plot_connectivity


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns





# 1. SET THE DIRECTORIES FOR SAVING FIGURES AND DATA

results_directory = './results/'

if not os.path.exists(results_directory):
    os.makedirs(results_directory)

data_directory = os.path.join(results_directory, 'Hopf_3node/data/')
figure_directory = os.path.join(results_directory, 'Hopf_3node/figures/')

if not os.path.exists(figure_directory):
   os.makedirs(figure_directory)

if not os.path.exists(data_directory):
    os.makedirs(data_directory)

# 2. INITIALISE CONNECTIVITY AND COUPLING

surrogate_connectivity = connectivity.Connectivity(number_of_regions=3,
                                                   number_of_connections=9,
                                                   weights=np.array([[2, 4, 0],
                                                                     [4, 2, 0],
                                                                     [0, 1, 1]]),
                                                   tract_lengths=np.array([[0, 10, 0],
                                                                           [10, 0, 3],
                                                                           [0, 3, 0]]),
                                                   region_labels=np.array(['DLPFC', 'IPSc', 'v1']),
                                                   centres=np.array([[0, 0, 0], [1, 1, 1]])
                                                   )

surrogate_connectivity.configure()

plot_connectivity(surrogate_connectivity, show_figure=False)


output_setting = simulator.monitors.SubSample(period=3.90625)
output_setting.configure()
simulation = simulator.Simulator(connectivity=surrogate_connectivity,
                                 coupling=coupling.Linear(),
                                 integrator=integrators.HeunStochastic(dt=2**-8, noise=noise.Additive()),
                                 model=models.oscillator.SupHopf(a=np.array([0.0])),
                                 monitors=[output_setting],
                                 simulation_length=5000
                                 )
simulation.configure()


# 3. CONFIGURE THE SIMULATION

def run_sim(global_coupling, noise):
    simulation.coupling.a = global_coupling
    simulation.integrator.noise.nsig = noise
    print("Starting supHopf simulation with coupling factor: " + str(global_coupling) + " noise: " + str(noise))
    results = simulation.run()
    time = results[0][0].squeeze() # <-- get the time
    data = results[0][1].squeeze() # <-- get the data
    return (global_coupling, noise, data, time) # <-- return the data and time


global_coupling_log = 10**np.r_[-5:0:25j] # <-- setting up the coupling range
noise_log = 10**np.r_[-6:-2:25j] # <-- setting up the noise range

# serial processing

# data = []
# for (ai, bi) in list(product(*[global_coupling_log, noise_log])):
#     data.append(run_sim(np.array([ai]), np.array([bi])))

# For multiprocessing

data = []

if __name__ == '__main__': # <-- this is needed for multiprocessing to work
    with Pool(processes=cpu_count()) as pool:
        for result in pool.starmap(run_sim, [(np.array([ai]), np.array([bi])) for (ai, bi) in list(product(*[global_coupling_log, noise_log]))]):
            data.append(result)


# 4. SAVE DATA AND PLOT FIGURES

for i in range(len(data)):
        sio.savemat(data_directory + 'Hopf_3node_gc-{0:02f}_noise-{1:02f}.mat'.format(float(data[i][0]), float(data[i][1])), {'data': data[i][2]})

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


