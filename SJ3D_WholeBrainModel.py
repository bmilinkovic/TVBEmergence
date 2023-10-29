#%%

import time
import os
from itertools import product
import datetime
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np
import scipy.io as sio
from scipy.stats import zscore
from tvb.simulator.lab import *

from plots.ssdigraphs import plot_connectivity, plot_wholebrain_connectivity


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# 1. SET THE DIRECTORIES FOR SAVING FIGURES AND DATA
results_directory = './results/'

if not os.path.exists(results_directory):
    os.makedirs(results_directory)


data_directory = os.path.join(results_directory, 'SJ3D_WB/data/')
figure_directory = os.path.join(results_directory, 'SJ3D_WB/figures/')

if not os.path.exists(figure_directory):
   os.makedirs(figure_directory)

if not os.path.exists(data_directory):
    os.makedirs(data_directory)


#%%
# 2. INITIALISE CONNECTIVITY AND COUPLING
conn = connectivity.Connectivity.from_file()
conn.configure()

output_monitor = monitors.TemporalAverage(period=3.90625)      # 256 Hz

simulation = simulator.Simulator(connectivity=conn,
                                 model=models.ReducedSetHindmarshRose(),
                                 coupling=coupling.Linear(),
                                 integrator=integrators.HeunStochastic(dt=2**-8, noise=noise.Additive()),
                                 monitors=[output_monitor],
                                 simulation_length=4000)  # 7 minutes.
simulation.configure()

plot_wholebrain_connectivity(conn, show_figure=False, annot=False)


now = datetime.datetime.now() # <-- get current date and time
matrix_size = conn.weights.shape # <-- get the size of the matrix

# Generate a unique identifier based on the matrix size and current date
identifier = f"{matrix_size[0]}x{matrix_size[1]}_{now.strftime('%Y-%m-%d_%H-%M')}"

# Use the identifier in the file name
plt.savefig(f"{figure_directory}/connectivity_{identifier}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{figure_directory}/connectivity_{identifier}.eps", format='eps', dpi=300, bbox_inches='tight')


#%%
# 3. CONFIGURE THE SIMULATION


def run_sim(global_coupling, noise):
    simulation.coupling.a = global_coupling
    simulation.integrator.noise.nsig = noise
    print("Starting SJ3D simulation with coupling factor: " + str(global_coupling) + " and noise: " + str(noise))
    results = simulation.run()
    time = results[0][0].squeeze()
    data = results[0][1].squeeze()
    data_cleaned = np.zeros([76, 1024])  # initialise structure for z-scored data.
    for i in range(len(conn.weights)):
        data_cleaned[i, :] = zscore(np.sum(data[:, 0, i, :], axis=1))
    return (global_coupling, noise, data_cleaned, time)

#%%
# for parameter sweep

global_coupling_log = 10**np.r_[-3:-0.11:8j] # <-- setting up the coupling range
noise_log = 10**np.r_[-3:-0.5:8j] # <-- setting up the noise range
# with connectivity

#%%
data_with_connectivity = []

if __name__ == '__main__': # <-- this is needed for multiprocessing to work
    with Pool(processes=cpu_count()) as pool:
        for result in pool.starmap(run_sim, [(np.array([ai]), np.array([bi])) for (ai, bi) in list(product(*[global_coupling_log, noise_log]))]):
            data_with_connectivity.append(result)


# without connectivity
data_with_noconnectivity = run_sim(np.array([0.000000000000000]), np.array([0.001]))


#%%
# 4. SAVE DATA AND PLOT FIGURES

for i in range(len(data_with_connectivity)):
        sio.savemat(data_directory + 'SJ3D_WholeBrainModel_gc-{0:02f}_noise-{1:02f}.mat'.format(float(data_with_connectivity[i][0]), float(data_with_connectivity[i][1])), {'data': data_with_connectivity[i][2]}) # <-- save the data as a matlab file



#sio.savemat(data_directory + 'SJ3D_WholeBrainModel_gc-{0:02f}.mat'.format(float(data_with_connectivity[0])), {'data': data_with_connectivity[1]})
sio.savemat(data_directory + 'SJ3D_WholeBrainModel_gc-{0:02f}.mat'.format(float(data_with_noconnectivity[0])), {'data': data_with_noconnectivity[1]})

#%% 
# 5. PLOT FIGURES

# Plotting for Connectivity

for i in range(len(data_with_connectivity)):
    fig, ax = plt.subplots()
    ax.set_title('Whole-brain with GC={0:02f} and Noise={1:02f}'.format(float(data_with_connectivity[0][0]), float(data_with_connectivity[0][1])), fontsize=10, fontname='Times New Roman', fontweight='bold')
    ax.set_xlabel('Time (ms)', fontsize=8, fontname='Times New Roman', fontweight='bold')
    ax.set_ylabel('Local Field Potential (LFP)', fontsize=8, fontname='Times New Roman', fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=7)
    right_side = ax.spines["right"]
    top_side = ax.spines["top"]
    right_side.set_visible(False)
    top_side.set_visible(False)
    ax.plot(data_with_connectivity[0][3], data_with_connectivity[i][2].T + 2*np.r_[:len(conn.weights)], linewidth=0.4)  # hacked the time because time is cumulative in the plots
    ax.axvspan(0, 500, alpha=0.5, color='grey')
    ax.set_yticks(2*np.r_[:len(conn.weights)]) # <-- set the ticks to be at the middle of each region
    ax.set_yticklabels(conn.region_labels)
    plt.savefig(f'{figure_directory}/SJ3D_WB_{data_with_connectivity[0][0]}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{figure_directory}/SJ3D_WB_{data_with_connectivity[0][0]}.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.show()

# Plotting for No Connectivity

fig, ax = plt.subplots()
ax.set_title('Whole-brain with GC={0:02f} and Noise={1:02f}'.format(float(data_with_connectivity[0]), float(data_with_connectivity[1])), fontsize=10, fontname='Times New Roman', fontweight='bold')
ax.set_xlabel('Time (ms)', fontsize=8, fontname='Times New Roman', fontweight='bold')
ax.set_ylabel('Local Field Potential (LFP)', fontsize=8, fontname='Times New Roman', fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=7)
right_side = ax.spines["right"]
top_side = ax.spines["top"]
right_side.set_visible(False)
top_side.set_visible(False)
ax.plot(data_with_connectivity[3], data_with_noconnectivity[2].T + 2*np.r_[:len(conn.weights)], linewidth=0.4)  # hacked the time because time is cumulative in the plots
ax.axvspan(0, 500, alpha=0.5, color='grey')
ax.set_yticks(2*np.r_[:len(conn.weights)]) # <-- set the ticks to be at the middle of each region
ax.set_yticklabels(conn.region_labels)
#ax.legend(['Node[1]', 'Node[2]', 'Node[3]', 'Node[4]', 'Node[5]'], loc='upper right', fontsize=6)
plt.savefig(f'{figure_directory}/SJ3D_WB_{data_with_noconnectivity[0][0]}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{figure_directory}/SJ3D_WB_{data_with_noconnectivity[0][0]}.eps', format='eps', dpi=300, bbox_inches='tight')
plt.show()








# %%
