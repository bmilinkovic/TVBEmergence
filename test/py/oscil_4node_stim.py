import random
import os

import numpy as np
import scipy as sc
import scipy.io as sio

import matplotlib as mpl
import matplotlib.pyplot as plt

from tvb.simulator.lab import *

# %%
# Setting results and figure directories
resultsDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/stimulation/data'
figureDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/stimulation/figures'

# %% Defining the stimulus

# setting connectivity
# set surrogate connectivity as a 'chain' on a spherical surface.
conn = connectivity.Connectivity()
conn.generate_surrogate_connectivity(4, motif='chain', undirected=False, these_centres='spherical')

plot_matrix(conn.weights, connectivity=conn)
plt.show()

# set weighting of stimulus coming into each node.
weighting = np.zeros((4,))
weighting[[0,1,2,3]] = 0.8

eqn_t = equations.PulseTrain()
eqn_t.parameters['onset'] = 2.5e3
eqn_t.parameters['T'] = 2000.0
eqn_t.parameters['tau'] = 100.0

# combine the spatial and temporal components into a StimuliRegion object
stimulus = patterns.StimuliRegion(
    temporal=eqn_t,
    connectivity=conn,
    weight=weighting)

# configure space and time of stimulus and plot it.
stimulus.configure_time(np.arange(0., 10e3, 2**-4))
stimulus.configure_space()

plot_pattern(stimulus)
plt.show()

# %%

ss_output_setting = simulator.monitors.TemporalAverage(period=3.90625)
ss_output_setting.configure()

sim = simulator.Simulator(model=models.Generic2dOscillator(),
                          connectivity=conn,
                          coupling=coupling.Linear(a=np.array([0.6])),
                          integrator=integrators.HeunStochastic(dt=2**-6, noise=noise.Additive(nsig=np.array([2**-4]))),
                          monitors=(ss_output_setting,),
                          stimulus=stimulus,
                          simulation_length=2000*57, # 2s of 56 trials minute simulation
                          ).configure()

(tavg_time, tavg_data),  = sim.run()

plt.plot(tavg_time, tavg_data[:, 0, :, 0], 'k', alpha=0.1)
plt.plot(tavg_time, tavg_data[:, 0, :, 0].mean(axis=1), 'r', alpha=1)
plt.show()

