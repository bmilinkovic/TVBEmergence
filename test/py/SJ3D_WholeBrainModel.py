import time
import os
from itertools import product

import pandas as pd
import numpy as np
import scipy.io as sio
from scipy.stats import zscore
from tvb.simulator.lab import *
from utils.pyutils.connMatrixPlotter import connMatrixPlotter

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# Directories for saving data and figures:
resultsDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/'
dataDir = os.path.join(resultsDir, 'SJ3D_WholeBrainModel/data/')
figureDir = os.path.join(resultsDir, 'SJ3D_WholeBrainModel/figures/')


if not os.path.exists(figureDir):
   os.makedirs(figureDir)

if not os.path.exists(dataDir):
    os.makedirs(dataDir)

# Connectivity Setting as Template Connectivity
conn = connectivity.Connectivity.from_file()
conn.configure()

output_monitor = monitors.TemporalAverage(period=3.90625)      # 256 Hz

simulation = simulator.Simulator(connectivity=conn,
                                 model=models.ReducedSetHindmarshRose(),
                                 coupling=coupling.Linear(),
                                 integrator=integrators.HeunStochastic(dt=2**-7, noise=noise.Additive(nsig=np.array([0.127]))),
                                 monitors=[output_monitor],
                                 simulation_length=4000)  # 7 minutes.
simulation.configure()


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
dataConnectivity = run_sim(np.array([2**-1.1]))

# without connectivity
dataNoConnectivity = run_sim(np.array([0.000000000000000]))

# %% Save Data


sio.savemat(dataDir + 'SJ3D_WholeBrainModel_gc-{0:02f}.mat'.format(float(dataConnectivity[0][0])), {'data': dataConnectivity[0][1]})
sio.savemat(dataDir + 'SJ3D_WholeBrainModel_gc-{0.02f}.mat'.format(float(dataNoConnectivity[0][0])), {'data': dataNoConnectivity[0][1]})

fileNameTemplate = r'/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/SJ3D_WholeBrainModel/figures/SJ3D_WholeBrainModel_gc-{0:02f}.svg'

# Plotting for Connectivity

fig, ax = plt.subplots()
ax.set_title('Whole-brain with GC={0:02f}'.format(float(dataConnectivity[0][0])), fontsize=10, fontname='Times New Roman', fontweight='bold')
ax.set_xlabel('Time (ms)', fontsize=8, fontname='Times New Roman', fontweight='bold')
ax.set_ylabel('Local Field Potential (LFP)', fontsize=8, fontname='Times New Roman', fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=7)
right_side = ax.spines["right"]
top_side = ax.spines["top"]
right_side.set_visible(False)
top_side.set_visible(False)
ax.plot(dataConnectivity[0][2], dataConnectivity[0][1], linewidth=0.4)  # hacked the time because time is cumulative in the plots
ax.axvspan(0, 500, alpha=0.5, color='grey')
#ax.legend(['Node[1]', 'Node[2]', 'Node[3]', 'Node[4]', 'Node[5]'], loc='upper right', fontsize=6)
plt.savefig(fileNameTemplate.format(float(dataConnectivity[0][0])), format='svg')
plt.clf()

# Plotting for No Connectivity

fig, ax = plt.subplots()
ax.set_title('5 Coupled SJ3D Models with GC={0:02f}'.format(float(dataNoConnectivity[0][0])), fontsize=10, fontname='Times New Roman', fontweight='bold')
ax.set_xlabel('Time (ms)', fontsize=8, fontname='Times New Roman', fontweight='bold')
ax.set_ylabel('Local Field Potential (LFP)', fontsize=8, fontname='Times New Roman', fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=7)
right_side = ax.spines["right"]
top_side = ax.spines["top"]
right_side.set_visible(False)
top_side.set_visible(False)
ax.plot(dataNoConnectivity[0][2], dataNoConnectivity[0][1], linewidth=0.4)  # hacked the time because time is cumulative in the plots
ax.axvspan(0, 500, alpha=0.5, color='grey')
#ax.legend(['Node[1]', 'Node[2]', 'Node[3]', 'Node[4]', 'Node[5]'], loc='upper right', fontsize=6)
plt.savefig(fileNameTemplate.format(float(dataNoConnectivity[0][0])), format='svg')
plt.clf()







