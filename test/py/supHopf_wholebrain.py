import time as tm
import numpy as np
import tvb.simulator.models.oscillator

from tvb.simulator.lab import *
from tvb.simulator.plot.phase_plane_interactive import PhasePlaneInteractive

import matplotlib as mpl

import matplotlib.pyplot as plt


#%%

# Create the God Damn mobel
hopf = tvb.simulator.models.oscillator.SupHopf()

# initialise the God Damn stoachistic integrator scheme.

heun = integrators.HeunDeterministic(dt=0.1)

# ppi_fig = PhasePlaneInteractive(model=hopf, integrator=heun)
# ppi_fig.show()

# Initialise the Connectivity.
con = connectivity.Connectivity.from_file('connectivity_66.zip')
nregions = len(con.region_labels)                               #number of regions
con.weights = con.weights - con.weights * np.eye((nregions))       #remove self-connections
con.weights = 0.2 * con.weights / np.abs(con.weights.max())     #scaled to a maximum value of 0.2
con.speed = np.array([sys.float_info.max])                      #set conduction speed (here we neglect it)
con.configure()

# Visualization.
plt.figure(figsize=(5,5))
plt.imshow(con.weights, interpolation='nearest', aspect='equal', cmap='jet')
plt.title('Structural Connectivity', fontsize=20)
plt.xticks(range(0, nregions), con.region_labels, fontsize=10, rotation=90)
plt.yticks(range(0, nregions), con.region_labels, fontsize=10)
cb=plt.colorbar(shrink=0.8)
cb.set_label('weights', fontsize=14)
plt.show()

# Initialise the Model.
mod = models.SupHopf()
mod.a = np.ones((nregions)) * (0)             #critical bifurcation point
mod.omega = np.ones((nregions)) * (2*np.pi*20.0) #omega= 2*pi*f

# Initialise a Coupling function.
coupl = 2.85  #coupling strength: G in equation above
con_coupling = coupling.Difference(a=np.array([coupl]))

# Integrator
dt = 0.01          #integration steps [ms]
sigma = 0.02       #standard deviation of the noise

nsigma = 2*np.sqrt(np.square(sigma)/2)
hiss = noise.Additive(nsig=np.array([nsigma, nsigma]))
heunint = integrators.HeunStochastic(dt=dt, noise=hiss)

# Initialise some Monitors with period in physical time.
mon_tavg = monitors.TemporalAverage(period=3.90625)

# Initialise the Simulator.
sim = simulator.Simulator(model=mod,
                          connectivity=con,
                          conduction_speed=np.float(con.speed),
                          coupling=con_coupling,
                          integrator=heunint,
                          monitors=[(mon_tavg)])
sim.configure()

# Perform simulation.
print("Starting simulation...")
tic = tm.time()
(t, y), = sim.run(simulation_length=20000)
print("Finished simulation.")
print('execute for ' + str(tm.time()-tic))


# Plot time series.
plt.figure(figsize=(10, 5))
plt.plot(t[:], y[:, 0, :, 0] + np.r_[:len(con.weights)], linewidth=0.4)
plt.title('Firing-rate time series', fontsize=20)
plt.xlabel('Time(s)', fontsize=18)
plt.yticks(np.arange(len(con.region_labels)), con.region_labels, fontsize=10)
plt.show()