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
from plots.ssdigraphs import plot_connectivity
import datetime

from multiprocessing import Pool, cpu_count, freeze_support



# 1. SET THE DIRECTORIES FOR SAVING FIGURES AND DATA

results_directory = './results/'

if not os.path.exists(results_directory):
    os.makedirs(results_directory)


data_directory = os.path.join(results_directory, 'SJ3D_3node/data/')
figure_directory = os.path.join(results_directory, 'SJ3D_3node/figures/')

if not os.path.exists(figure_directory):
   os.makedirs(figure_directory)

if not os.path.exists(data_directory):
    os.makedirs(data_directory)

# 2. INITIALISE CONNECTIVITY AND COUPLING

default = connectivity.Connectivity.from_file()
default.configure()

# get a row and columnar index for specific regions or nodes
idx = np.r_[random.sample(range(0, 75), 3)]

# configure weights structural and connectivity
changedWeights = np.array([[2, 4, 0],
                           [4, 2, 0],
                           [0, 1, 1]])

tracts_nodelay = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

subset = connectivity.Connectivity(weights=changedWeights,
                                           tract_lengths=default.tract_lengths[idx][:, idx],
                                           centres=default.centres[idx],
                                           region_labels=default.region_labels[idx])
subset.configure()


plot_connectivity(subset, show_figure=False)
# plt.savefig(figure_directory + 'structural_connectivity.png')


now = datetime.datetime.now() # <-- get current date and time
matrix_size = subset.weights.shape # <-- get the size of the matrix

# Generate a unique identifier based on the matrix size and current date
identifier = f"{matrix_size[0]}x{matrix_size[1]}_{now.strftime('%Y-%m-%d_%H-%M')}"

# Use the identifier in the file name
plt.savefig(f"{figure_directory}/connectivity_{identifier}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{figure_directory}/connectivity_{identifier}.eps", format='eps', dpi=300, bbox_inches='tight')

# 3. INITIALISE SIMULATOR

# configure monitors
monitors = monitors.TemporalAverage(period=3.90625) # <-- configure the temporal average monitor

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
    simulation.coupling.a = global_coupling # <-- set the coupling
    simulation.integrator.noise.nsig = noise # <-- set the noise
    print("Starting SJ3D simulation with coupling factor: " + str(global_coupling) + " and noise: " + str(noise)) # <-- print the coupling and noise
    results = simulation.run() # <-- run the simulation
    time = results[0][0].squeeze() # <-- extract the time
    data = results[0][1].squeeze() # <-- extract the data
    data_cleaned = np.zeros([3, 1024])  # initialise structure for z-scored data.
    for i in range(len(changedWeights)): # <-- loop over the number of nodes
        data_cleaned[i] = zscore(np.sum(data[:, 0, i, :], axis=1)) # <-- z-score the data
    return (global_coupling, noise, data_cleaned, time) # <-- return the coupling, noise, data and time


global_coupling_log = 10**np.r_[-2:-0.02:25j] # <-- set the coupling range
noise_log = 10**np.r_[-3:0.0:25j] # <-- set the noise range

# for sequential processing

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

# save data
for i in range(len(data)):
        sio.savemat(data_directory + 'SJ3D_3node_withlink_gc-{0:02f}_noise-{1:02f}.mat'.format(float(data[i][0]), float(data[i][1])), {'data': data[i][2]})


# plotting figures
for i in range(len(data)):
    fig, ax = plt.subplots()
    ax.set_title('3 coupled SJ3D NMMs with GC={0:02f} and Noise={1:02f}'.format(float(data[i][0]), float(data[i][1])), fontsize=10, fontname='Times New Roman', fontweight='bold')
    ax.set_xlabel('Time (ms)', fontsize=8, fontname='Times New Roman', fontweight='bold') # <-- set the x-axis label
    ax.set_ylabel('Local Field Potential (LFP)', fontsize=8, fontname='Times New Roman', fontweight='bold') # <-- set the y-axis label
    ax.tick_params(axis='both', which='major', labelsize=7) # <-- set the tick label size
    right_side = ax.spines["right"] # <-- remove the top and right spines
    top_side = ax.spines["top"] # <-- remove the top and right spines
    right_side.set_visible(False) # <-- remove the top and right spines
    top_side.set_visible(False) # <-- remove the top and right spines
    ax.plot(data[0][3], data[i][2].T + np.r_[:len(subset.weights)], linewidth=0.4)  # hacked the time because time is cumulative in the plots
    ax.axvspan(0, 500, alpha=0.5, color='grey') # <-- this is the grey box
    ax.legend(subset.region_labels, loc='upper right', fontsize=6) # <-- set the legend to be the region labels
    ax.set_yticks(np.r_[:len(subset.weights)]) # <-- set the ticks to be at the middle of each region
    ax.set_yticklabels(subset.region_labels) # <-- set the tick labels to be the region labels
    plt.savefig(f"{figure_directory}/SJ3D_3node_gc-{data[i][0]}_noise-{data[i][1]}.png", dpi=300, bbox_inches='tight') # <-- save the figure
    plt.savefig(f"{figure_directory}/SJ3D_3node_gc-{data[i][0]}_noise-{data[i][1]}.eps", format='eps', dpi=300, bbox_inches='tight') # <-- save the figure
    plt.clf()

 
                
