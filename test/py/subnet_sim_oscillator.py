from tvb.simulator.lab import *
import numpy as np
import matplotlib.pyplot as plt
import utils.pyutils.connMatrixPlotter
from networks.pynetworks.subset9Modular36 import subnet9mod36
import time
import scipy.io as sio

### Saving figures setup ###
### if we want to not save, just set save = False ###
figureDir = "/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/figures/"
savefig = True
###########################

### Saving results setup ###
### if we want to not save, just set savemat = False ###
dataDir = "/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/data/"
resultsmat = True
###########################


conn = subnet9mod36()
monitors = simulator.monitors.SubSample(period=1.953125)
monitors.configure()

# Set up the simulation with all the correct arguments.
simulation = simulator.Simulator(connectivity=conn,
                                 coupling=coupling.Linear(),
                                 integrator=integrators.HeunStochastic(dt=0.05, noise=noise.Additive(nsig = np.array([0.004]))),
                                 model=models.Generic2dOscillator(),
                                 monitors=[monitors],
                                 simulation_length=2000)
simulation.configure()


# Defining a function to run a simulation
def run_sim(global_coupling):
    simulation.coupling.a = global_coupling
    print("Starting Gen2dOscillator simulation with coupling factor " + str(global_coupling))
    results = simulation.run()
    data = results[0][1].squeeze()
    return (global_coupling, data)

# running a parameter sweep of the global coupling parameter
gc_range = np.arange(0.0, 5.1, .50)
data = []
for gc in gc_range:
    data.append((run_sim(np.array([gc]))))

#plot connectivity

plt.figure()
utils.pyutils.connMatrixPlotter.connMatrixPlotter(conn)
f1 = plt.gcf()
plt.show()

# This below needs some work.


if resultsmat:
for i in range(len(data)):
    if os.path.exists(dataDir + 'oscSim.mat'):
        sio.savemat(dataDir + 'oscSim_{0}_{1}.mat'.format(i, int(time.time())), {'data': data[i][1]})



if savefig:
    if os.path.exists(figureDir + 'oscFig.png'):
        plt.savefig(figureDir + 'oscFig_{}.png'.format(int(time.time())))
    else:
        plt.savefig(figureDir + 'oscFig.png')

