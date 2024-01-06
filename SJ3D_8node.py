#%%
import numpy as np
import random
from tvb.simulator.lab import *
#from tvb.analyzers import compute_proxy_metastability

from scipy.stats import zscore
import scipy.io as sio
import pandas as pd
import os
import datetime
from multiprocessing import Pool, cpu_count

import matplotlib as mpl
import matplotlib.pyplot as plt

from itertools import product

from plots.ssdigraphs import plot_connectivity


#%% 1. SET THE DIRECTORIES FOR SAVING FIGURES AND DATA

results_directory = './results/'

if not os.path.exists(results_directory):
    os.makedirs(results_directory)


data_directory = os.path.join(results_directory, 'SJ3D_8node/data/')
figure_directory = os.path.join(results_directory, 'SJ3D_8node/figures/')

if not os.path.exists(figure_directory):
   os.makedirs(figure_directory)

if not os.path.exists(data_directory):
    os.makedirs(data_directory)

#%% 2. INITIALISE CONNECTIVITY AND COUPLING 
default = connectivity.Connectivity.from_file()
default.configure()


idx = np.r_[random.sample(range(0, 75), 8)]         # <-- get a row and columnar index for specific regions or nodes


changed_weights = np.array([[0, 0, 0, 0, 0],         # <-- configure weights and tract lengths for the subset of nodes
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])

tracts_nodelay = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])

subset = connectivity.Connectivity(weights=default.weights[idx][:, idx],
                                           tract_lengths=default.tract_lengths[idx][:, idx],
                                           centres=default.centres[idx],
                                           region_labels=default.region_labels[idx])
subset.configure()                                  # <-- configure the subset connectome


plot_connectivity(subset, show_figure=False, annot=True)         # <-- plot the subset connectome



#%%

now = datetime.datetime.now() # <-- get current date and time
matrix_size = subset.weights.shape # <-- get the size of the matrix

# Generate a unique identifier based on the matrix size and current date
identifier = f"{matrix_size[0]}x{matrix_size[1]}_{now.strftime('%Y-%m-%d_%H-%M')}"

# Use the identifier in the file name
plt.savefig(f"{figure_directory}/connectivity_{identifier}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{figure_directory}/connectivity_{identifier}.svg", format='svg', dpi=300, bbox_inches='tight')

#%% 3. CONFIGURE THE SIMULATION

monitors = monitors.TemporalAverage(period=3.90625)         # <-- configure the temporal average monitor    

simulation = simulator.Simulator(connectivity=subset,
                                 coupling=coupling.Linear(),
                                 integrator=integrators.HeunStochastic(dt=2**-6,
                                                                       noise=noise.Additive()),
                                 monitors=[monitors],
                                 model=models.ReducedSetHindmarshRose(),
                                 simulation_length=4000)
simulation.configure()                                      # <-- configure the simulation



#%% 4. RUN THE SIMULATION

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
    data_cleaned = np.zeros([8, 1024])  # initialise structure for z-scored data.
    for i in range(len(subset.weights)):
        data_cleaned[i] = zscore(np.sum(data[:, 0, i, :], axis=1))
    return (global_coupling, noise, data_cleaned, time)

# global_coupling_log = [0.3, 0.7]
# noise_log = [0.01, 0.07]

global_coupling_log = 10**np.r_[-2:-0.02:20j] # <-- set the coupling range -2:-0.5:25j
noise_log = 10**np.r_[-3:-0.0:20j] # <-- set the noise range -3:-0.002:25j

# serial processing
# data = []
# for (ai, bi) in list(product(*[global_coupling_log, noise_log])):
#     data.append(run_sim(np.array([ai]), np.array([bi])))

# multiprocessing

data = []
#data = run_sim(np.array([global_coupling_log]), np.array([noise_log]))

if __name__ == '__main__': # <-- this is needed for multiprocessing to work
    with Pool(processes=cpu_count()-1) as pool:
        for result in pool.starmap(run_sim, [(np.array([ai]), np.array([bi])) for (ai, bi) in list(product(*[global_coupling_log, noise_log]))]):
            data.append(result)

#%% 5. SAVE THE DATA AND PLOT THE FIGURES

if not os.path.exists(os.path.join(data_directory, 'py')): # <-- create a directory for the python data
    os.makedirs(os.path.join(data_directory, 'py'))

#np.save(os.path.join(data_directory, 'py', 'data.npy'), data) # <-- save the data as a numpy file

for i in range(len(data)):
        sio.savemat(data_directory + 'SJ3D_8node_gc-{0:02f}_noise-{1:02f}.mat'.format(float(data[i][0]), float(data[i][1])), {'data': data[i][2]}) # <-- save the data as a matlab file


## Load the data and plot the figures
# data = []
# for filename in os.listdir(os.path.join(data_directory, 'py')):
#     if filename.endswith('.npy'):
#         data.append(np.load(os.path.join(data_directory, 'py', filename)))

#%% PLOT DATA

for i in range(len(data)):
    fig, ax = plt.subplots()
    ax.set_title('8 SJ3D Models with GC={0:02f} and Noise={1:02f}'.format(float(data[i][0]), float(data[i][1])), fontsize=10, fontname='Times New Roman', fontweight='bold')
    ax.set_xlabel('Time (ms)', fontsize=8, fontname='Times New Roman', fontweight='bold')
    ax.set_ylabel('Local Field Potential (LFP)', fontsize=8, fontname='Times New Roman', fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=7)
    right_side = ax.spines["right"]
    top_side = ax.spines["top"]
    right_side.set_visible(False)
    top_side.set_visible(False)
    ax.plot(data[0][3], data[i][2].T + 2*np.r_[:len(subset.weights)], linewidth=0.4)  # hacked the time because time is cumulative in the plots
    ax.axvspan(0, 500, alpha=0.5, color='grey') # <-- this is the grey box
    #ax.legend(subset.region_labels, loc='upper right', fontsize=6) # <-- set the legend to be the region labels
    ax.set_yticks(2*np.r_[:len(subset.weights)]) # <-- set the ticks to be at the middle of each region
    ax.set_yticklabels(subset.region_labels) # <-- set the tick labels to be the region labels
    plt.savefig(f"{figure_directory}/SJ3D_8node_gc-{data[i][0]}_noise-{data[i][1]}.png", dpi=300, bbox_inches='tight') # <-- save the figure
    plt.savefig(f"{figure_directory}/SJ3D_8node_gc-{data[i][0]}_noise-{data[i][1]}.svg", format='svg', dpi=300, bbox_inches='tight') # <-- save the figure
    plt.clf()






# %%
