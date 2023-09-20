import time
import os
from itertools import product
import datetime

import pandas as pd
import numpy as np
import scipy.io as sio
from scipy.stats import zscore
from tvb.simulator.lab import *

from plots.ssdigraphs import plot_connectivity, plot_wholebrain_connectivity


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

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


# 2. INITIALISE CONNECTIVITY AND COUPLING
conn = connectivity.Connectivity.from_file()
conn.configure()

output_monitor = monitors.TemporalAverage(period=3.90625)      # 256 Hz

simulation = simulator.Simulator(connectivity=conn,
                                 model=models.ReducedSetHindmarshRose(),
                                 coupling=coupling.Linear(),
                                 integrator=integrators.HeunStochastic(dt=2**-10, noise=noise.Additive(nsig=np.array([0.001]))),
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



# 3. CONFIGURE THE SIMULATION


def run_sim(global_coupling):
    simulation.coupling.a = global_coupling
    print("Starting SJ3D simulation with coupling factor: " + str(global_coupling))
    results = simulation.run()
    time = results[0][0].squeeze()
    data = results[0][1].squeeze()
    data_cleaned = np.zeros([76, 1024])  # initialise structure for z-scored data.
    for i in range(len(conn.weights)):
        data_cleaned[i, :] = zscore(np.sum(data[:, 0, i, :], axis=1))
    return (global_coupling, data_cleaned, time)




# with connectivity
data_with_connectivity = run_sim(np.array([2**-2.1]))

# without connectivity
data_with_noconnectivity = run_sim(np.array([0.000000000000000]))

# 4. SAVE DATA AND PLOT FIGURES


sio.savemat(data_directory + 'SJ3D_WholeBrainModel_gc-{0:02f}.mat'.format(float(data_with_connectivity[0][0])), {'data': data_with_connectivity[0][1]})
sio.savemat(data_directory + 'SJ3D_WholeBrainModel_gc-{0:02f}.mat'.format(float(data_with_noconnectivity[0][0])), {'data': data_with_noconnectivity[0][1]})

# 5. PLOT FIGURES

# Plotting for Connectivity

fig, ax = plt.subplots()
ax.set_title('Whole-brain with GC={0:02f}'.format(float(data_with_connectivity[0][0])), fontsize=10, fontname='Times New Roman', fontweight='bold')
ax.set_xlabel('Time (ms)', fontsize=8, fontname='Times New Roman', fontweight='bold')
ax.set_ylabel('Local Field Potential (LFP)', fontsize=8, fontname='Times New Roman', fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=7)
right_side = ax.spines["right"]
top_side = ax.spines["top"]
right_side.set_visible(False)
top_side.set_visible(False)
ax.plot(data_with_connectivity[0][2], data_with_connectivity[0][1] + np.r_[:len(conn.weights)], linewidth=0.4)  # hacked the time because time is cumulative in the plots
ax.axvspan(0, 500, alpha=0.5, color='grey')
#ax.legend(['Node[1]', 'Node[2]', 'Node[3]', 'Node[4]', 'Node[5]'], loc='upper right', fontsize=6)
plt.savefig(f'{figure_directory}/SJ3D_WB_{data_with_connectivity[0][0]}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{figure_directory}/SJ3D_WB_{data_with_connectivity[0][0]}.eps', format='eps', dpi=300, bbox_inches='tight')
plt.clf()

# Plotting for No Connectivity

fig, ax = plt.subplots()
ax.set_title('5 Coupled SJ3D Models with GC={0:02f}'.format(float(data_with_noconnectivity[0][0])), fontsize=10, fontname='Times New Roman', fontweight='bold')
ax.set_xlabel('Time (ms)', fontsize=8, fontname='Times New Roman', fontweight='bold')
ax.set_ylabel('Local Field Potential (LFP)', fontsize=8, fontname='Times New Roman', fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=7)
right_side = ax.spines["right"]
top_side = ax.spines["top"]
right_side.set_visible(False)
top_side.set_visible(False)
ax.plot(data_with_noconnectivity[0][2], data_with_noconnectivity[0][1] + np.r_[:len(conn.weights)], linewidth=0.4)  # hacked the time because time is cumulative in the plots
ax.axvspan(0, 500, alpha=0.5, color='grey')
#ax.legend(['Node[1]', 'Node[2]', 'Node[3]', 'Node[4]', 'Node[5]'], loc='upper right', fontsize=6)
plt.savefig(f'{figure_directory}/SJ3D_WB_{data_with_noconnectivity[0][0]}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{figure_directory}/SJ3D_WB_{data_with_noconnectivity[0][0]}.eps', format='eps', dpi=300, bbox_inches='tight')
plt.clf()







