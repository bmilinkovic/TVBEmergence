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

# Set the directories for saving figures and data
figureDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/figures/osc2d_3node/'
dataDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/data/osc2d_3node/'

# Create simulation

surrogate_connectivity = connectivity.Connectivity(number_of_regions=3,
                                                   number_of_connections=9,
                                                   weights=np.array([[0, 15, 0], [1, 0, 1], [0, 15, 0]]),
                                                   tract_lengths=np.array([[0, 10, 0], [10, 0, 3], [0, 3, 0]]),
                                                   region_labels=np.array(['DLPFC', 'IPSc', 'v1']),
                                                   centres=np.array([[0, 0, 0], [1, 1, 1]])
                                                   )
connMatrixPlotter(surrogate_connectivity)


output_setting = simulator.monitors.SubSample(period=3.90625)
output_setting.configure()
simulation = simulator.Simulator(connectivity=surrogate_connectivity,
                                 coupling=coupling.Linear(),
                                 integrator=integrators.HeunStochastic(dt=0.5, noise=noise.Additive(nsig=np.array([0.01]))),
                                 model=models.Generic2dOscillator(a=np.array([2.2]),
                                                                  b=np.array([-1.0]),
                                                                  c=np.array([0.0]),
                                                                  d=np.array([0.1]),
                                                                  I=np.array([0.0]),
                                                                  alpha=np.array([1.0]),
                                                                  beta=np.array([0.2]),
                                                                  gamma=np.array([-1.0]),
                                                                  e=np.array([0.0]),
                                                                  g=np.array([1.0]),
                                                                  f=np.array([0.333]),
                                                                  tau=np.array([1.25])),
                                 monitors=[output_setting],
                                 simulation_length=5000
                                 )
simulation.configure()

results = simulation.run()
time = results[0][0].squeeze()
data = results[0][1].squeeze()



