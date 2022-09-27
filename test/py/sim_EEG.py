import os
import time
from itertools import product

import numpy as np
import scipy as sc
import scipy.io as sio

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from tvb.simulator.lab import *
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.projections import ProjectionMatrix, ProjectionSurfaceEEG
from tvb.datatypes.sensors import SensorsEEG

model = models.Generic2dOscillator(a=np.array([0.1]), tau=np.array([2.0]))

conn = connectivity.Connectivity.from_file('connectivity_192.zip')
conn.speed = np.array([4.0])
conn_coupling = coupling.Difference(a=np.array([0.014]))

region_map = RegionMapping.from_file('regionMapping_16k_192.txt')
sensorsEEG = SensorsEEG.from_file('eeg_unitvector_62.txt.bz2')
projectionEEG = ProjectionSurfaceEEG.from_file('projection_eeg_62_surface_16k.mat', matlab_data_name="ProjectionMatrix")

integ = integrators.HeunStochastic(dt=2**-4,
                                   noise=noise.Additive(nsig=np.array([2**-5,])))
freq_samp = 1e3/1024.0

monitorMEG=monitors.MEG.from_file()
monitorMEG.period=freq_samp
mons = (monitors.EEG(sensors=sensorsEEG, projection=projectionEEG, region_mapping=region_map, period=freq_samp),
        monitorMEG,
        monitors.iEEG.from_file('seeg_588.txt', 'projection_seeg_588_surface_16k.npy', period=freq_samp),
        monitors.ProgressLogger(period=100.0),)

local_coupling_strength = np.array([2**-10])
default_cortex = Cortex.from_file(region_mapping_file='regionMapping_16k_192.txt')
default_cortex.region_mapping_data.connectivity = conn
default_cortex.coupling_strength = local_coupling_strength

sim = simulator.Simulator(model=model,
                          connectivity=conn,
                          coupling=conn_coupling,
                          integrator=integ,
                          monitors=mons,
                          surface=default_cortex,
                          simulation_length=1000.0)
sim.configure()

eeg, meg, seeg, _ = sim.run()

plt.figure()
for i, mon, in enumerate((eeg, meg, seeg)):
    subplot(3, 1, i + 1)
    time, data = mon
    plot(time, data[:,0,:,0], 'k', alpha=0.1)
    ylabel(['EEG','MEG','sEEG'][i])

