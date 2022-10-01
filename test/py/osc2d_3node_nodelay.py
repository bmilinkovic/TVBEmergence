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
resultsDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/'
dataDir = os.path.join(resultsDir, 'osc2d_3node_UNCOUPLED_nodelay_a-1.0_ps_gc-noise/data/')
figureDir = os.path.join(resultsDir, 'osc2d_3node_UNCOUPLED_nodelay_a-1.0_ps_gc-noise/figures/')


if not os.path.exists(figureDir):
   os.makedirs(figureDir)

if not os.path.exists(dataDir):
    os.makedirs(dataDir)

# Create simulation

surrogate_connectivity = connectivity.Connectivity(number_of_regions=3,
                                                   number_of_connections=9,
                                                   speed=np.array([sys.float_info.max]),
                                                   weights=np.array([[0, 0, 0],
                                                                     [0, 0, 0],
                                                                     [0, 0, 0]]),
                                                   tract_lengths=np.array([[0, 0, 0],
                                                                           [0, 0, 0],
                                                                           [0, 0, 0]]),
                                                   region_labels=np.array(['DLPFC', 'IPSc', 'V1']),
                                                   centres=np.array([[0, 0, 0], [1, 1, 1]])
                                                   )
surrogate_connectivity.configure()

connMatrixPlotter(surrogate_connectivity)

#raw_output_setting = simulator.monitors.Raw()
ss_output_setting = simulator.monitors.TemporalAverage(period=3.90625)
ss_output_setting.configure()
simulation = simulator.Simulator(connectivity=surrogate_connectivity,
                                 coupling=coupling.Linear(),
                                 integrator=integrators.HeunStochastic(dt=0.5, noise=noise.Additive()),
                                 model=models.Generic2dOscillator(a=np.array([1.0]),
                                                                  b=np.array([-1.0]),
                                                                  c=np.array([0.0]),
                                                                  d=np.array([0.1]),
                                                                  I=np.array([0.0]),
                                                                  alpha=np.array([1.0]),
                                                                  beta=np.array([0.2]),
                                                                  gamma=np.array([1.0]), # you can set this to anti-phase coupling by setting gamma = -1.0
                                                                  e=np.array([0.0]),
                                                                  g=np.array([1.0]),
                                                                  f=np.array([0.333]),
                                                                  tau=np.array([1.25])),
                                 monitors=[ss_output_setting],
                                 simulation_length=4000
                                 )
simulation.configure()

def run_sim(global_coupling, noise):
    simulation.coupling.a = global_coupling
    simulation.integrator.noise.nsig = noise
    print("Starting Generic2dOscillator simulation with coupling factor: " + str(global_coupling) + " and noise: " + str(noise))
    results = simulation.run()
    time = results[0][0].squeeze()
    data = results[0][1].squeeze()
    return (global_coupling, noise, data, time)


#global_coupling = np.r_[0.3:0.7:0.1]
global_coupling_log = 10**np.r_[-5:0:20j]
#noise = np.r_[0:0.04:0.01]
noise_log = 10**np.r_[-6:-2:20j]
#conduction_speed = np.r_[0:21:5]

data = []
for (ai, bi) in list(product(*[global_coupling_log, noise_log])):
    data.append(run_sim(np.array([ai]), np.array([bi])))

# global_coupling = np.r_[0.0:0.9:0.1]
# data = []
# for gc in global_coupling:
#     data.append((run_sim(np.array([gc]))))

# # Saving the data files
for i in range(len(data)):
        sio.savemat(dataDir + 'osc2d_3node_nodelay_gc-{0:02f}_noise-{1:02f}.mat'.format(float(data[i][0]), float(data[i][1])), {'data': data[i][2]})


fileNameTemplate = r'/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/osc2d_3node_UNCOUPLED_nodelay_a-1.0_ps_gc-noise/figures/osc2d_3node_nodelay_gc-{0:02f}_noise-{1:02f}.svg'
for i in range(len(data)):
    fig, ax = plt.subplots()
    ax.set_title('3 Coupled 2D-Oscillators with GC={0:02f} and Noise={1:02f}'.format(float(data[i][0]), float(data[i][1])), fontsize=10, fontname='Times New Roman', fontweight='bold')
    ax.set_xlabel('Time (ms)', fontsize=8, fontname='Times New Roman', fontweight='bold')
    ax.set_ylabel('Value of Fast Variable (V)', fontsize=8, fontname='Times New Roman', fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=7)
    right_side = ax.spines["right"]
    top_side = ax.spines["top"]
    right_side.set_visible(False)
    top_side.set_visible(False)
    ax.plot(data[0][3], data[i][2], linewidth=0.4)  # hacked the time because time is cumulative in the plots
    ax.axvspan(0, 500, alpha=0.5, color='grey')
    ax.legend(['DLPFC', 'IPSc', 'V1'], loc='upper right', fontsize=6)
    plt.savefig(fileNameTemplate.format(float(data[i][0]), float(data[i][1])), format='svg')
    plt.clf()



# TRYING TO CALCULATION
# from statsmodels.tsa.stattools import adfuller
#
# result = adfuller(data[0][1][:,0])
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# print('Critical Values:')
# for key, value in result[4].items():
# 	print('\t%s: %.3f' % (key, value))