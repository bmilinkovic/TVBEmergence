# %% IMPORT LIBRARIES

import os
import os.path
import sys
import errno
import time
import timeit
import numpy as np

from scipy.optimize import fsolve

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors, cm

from tvb.simulator.lab import *

# %% Intialise particular things

cwd = os.getcwd()

# if you have SC data set their directories

# project_dir = cwd + '/ExperimentalData/'
# results_dirt + cwd + '/ResultsOutput/'

# %% Structural connectivity Matrix

conn = connectivity.Connectivity.from_file()

# normalise connectivity always

conn.weights = conn.weights / np.max(conn.weights)
n_regions = len(conn.region_labels)

# %% plot connectivity

plt.figure(figsize=(20, 10))
plt.subplot(121)
plt.imshow(conn.weights, interpolation='nearest', cmap='jet')
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Normalised Default SC')

plt.subplot(122)
norm = colors.LogNorm(1e-7, conn.weights.max())
im = plt.imshow(conn.weights, norm=norm, cmap=cm.jet)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.gca().set_title('Normalized SC log scale', fontsize=13.0)

plt.show()

# %% Chosen REGIONS for Epileptogenic and Propogration areas

hz_val = -2.4
pz_val = -1.9
ez_val = -1.6

ez_idx = np.array([6, 34], dtype=np.int32)
pz_idx = np.array([5, 11, 27], dtype=np.int32)

# %% Epileptor model setup.

epileptor = models.Epileptor(variables_of_interest=['x1', 'y1', 'z', 'x2', 'y2', 'g', 'x2 - x1'])

epileptor.r = np.array([1.0 / 2857])
epileptor.Ks = np.ones(n_regions) * (-1.0)
epileptor.tt = np.array([1.0])

simLen = 10000

epileptor.x0 = np.ones(n_regions) * hz_val
epileptor.x0[ez_idx] = ez_val
epileptor.x0[pz_idx] = pz_val

# %% COUPLING and other stuff

coupling = coupling.Difference(a=np.array([1.0]))

mon_tavg = monitors.TemporalAverage(period=3.90625)
# %% SETTING THE NOISE

noiseON = True

nsf = 5.0  # noise-scaling factor, can be used later for para sweep.

hiss = noise.Additive(nsig=nsf * np.array([0.01, 0.01, 0., 0.00015, 0.00015, 0.]))

if (noiseON):
    heunint = integrators.HeunStochastic(dt=0.04, noise=hiss)
else:
    heunint = integrators.HeunDeterministic(dt=0.04)


# %% Find fixed point to initialise the damn epileptor in stable position.

def get_equilibrium(model, init):
    nvars = len(model.state_variables)
    cvars = len(model.cvar)

    def func(x):
        fx = model.dfun(x.reshape((nvars, 1, 1)),
                        np.zeros((cvars, 1, 1)))
        return fx.flatten()

    x = fsolve(func, init)
    return x


epileptor_equil = models.Epileptor()
epileptor_equil.x0 = np.array([-3.0])
init_cond = get_equilibrium(epileptor_equil, np.array([0.0, 0.0, 3.0, -1.0, 1.0, 0.0]))
init_cond_reshaped = np.repeat(init_cond, n_regions).reshape((1, len(init_cond), n_regions, 1))

# %% SIM SET UP

sim = simulator.Simulator(model=epileptor,
                          initial_conditions=init_cond_reshaped,
                          connectivity=conn,
                          coupling=coupling,
                          conduction_speed=np.inf,
                          integrator=heunint,
                          monitors=[mon_tavg])

sim.configure()

# %% RUN SIMULATIONN!

[(ttavg, tavg)] = sim.run(simulation_length=simLen)

# check simulation

time_steps = ttavg
X = tavg[:, 0, :, 0]
Z = tavg[:, 2, :, 0]

nn = np.r_[0:X.shape[1]]

# plot source space activity

plt.figure(figsize=(15, 20))
plt.plot(time_steps, X + np.r_[0:len(nn)] + 2, 'r')
plt.yticks(np.r_[0:len(nn)], np.r_[0:len(nn)], fontsize=10)
plt.title("Source signal (x)", fontsize=15)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Region', fontsize=12)
# plt.savefig(os.path.join(results_dir,"VEP_SL_patient1.png"))
plt.show()

# %% LFP simulation

LFP = tavg[:, 0, :, 0] + tavg[:, 3, :, 0]

plt.figure(figsize=(15, 20))
plt.plot(time_steps, LFP + np.r_[0:len(nn)] + 2, 'r')
plt.yticks(np.r_[0:len(nn)], np.r_[0:len(nn)], fontsize=10)
plt.title("Source signal (x)", fontsize=15)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Region', fontsize=12)
# plt.savefig(os.path.join(results_dir,"VEP_LPF_patient1.png"))
plt.show()

# %% now plotting just the regions

Nregions = np.r_[0:len(nn)]

fidx = 500
lidx = 1000

y = X[fidx:lidx, :].T

eni = np.array([0, 5, 6, 11, 27, 34])

plt.figure(figsize=(12, 8))
for i, yi in enumerate(y[eni]):
    if eni[i] in ez_idx:
        plt.plot(yi + 2 * i - 0.5, 'r', linewidth=1.5)
    elif eni[i] in pz_idx:
        plt.plot(yi + 2 * i - 0.25, 'y', linewidth=1.5)
    else:
        plt.plot(yi + 2 * i, 'g', linewidth=2)

plt.ylabel('#Nodes', fontsize=14)
plt.xticks(np.arange(0, 400, 100), 0.1 * (np.arange(0, 400, 100)), fontsize=10, rotation='horizontal')
plt.yticks(2 * np.r_[:len(eni)] - 2, [Nregions[i] for i in eni])
# plt.savefig(os.path.join(results_dir, "VEP_SL_ROIs_patient1.png"))
plt.show()

# plot LFP

Nregions = np.r_[0:len(nn)]

fidx = 500
lidx = 1000

y = LFP[fidx:lidx, :].T

eni = np.array([0, 5, 6, 11, 27, 34])

plt.figure(figsize=(12, 8))
for i, yi in enumerate(y[eni]):
    if eni[i] in ez_idx:
        plt.plot(yi + 2 * i - 0.5, 'r', linewidth=1.5)
    elif eni[i] in pz_idx:
        plt.plot(yi + 2 * i - 0.25, 'y', linewidth=1.5)
    else:
        plt.plot(yi + 2 * i, 'g', linewidth=2)

plt.ylabel('#Nodes', fontsize=14)
plt.xticks(np.arange(0, 400, 100), 0.1 * (np.arange(0, 400, 100)), fontsize=10, rotation='horizontal')
plt.yticks(2 * np.r_[:len(eni)] - 2, [Nregions[i] for i in eni])
# plt.savefig(os.path.join(results_dir, "VEP_SL_ROIs_patient1.png"))
plt.show()

# PLOT NODES TOGETHER:

plt.figure(figsize=(18, 6))
for roi in ez_idx:
    plt.plot(ttavg[fidx:lidx], X[fidx:lidx, roi], label='Node' + str(roi) + '(EZ)', color='r', alpha=0.7, linewidth=2)
for roi in pz_idx:
    plt.plot(ttavg[fidx:lidx], X[fidx:lidx, roi], label='Node' + str(roi) + '(PZ)', color='y', alpha=0.7, linewidth=2)
for roi in np.array([1, 2]):
    plt.plot(ttavg[fidx:lidx], X[fidx:lidx, roi], label='Node' + str(roi) + '(HZ)', color='g', alpha=0.7, linewidth=2)

plt.title("Source signal (x)", fontsize=15)
plt.xlabel('Time', fontsize=12)
plt.legend()
plt.tight_layout()
# plt.savefig(os.path.join(results_dir,"VEP_fastvariable_ROIs_patient1.png"))
plt.show()
