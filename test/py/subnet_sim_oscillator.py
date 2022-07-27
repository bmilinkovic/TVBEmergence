from tvb.simulator.lab import *
import numpy as np
import matplotlib.pyplot as plt
import utils.pyutils.connMatrixPlotter
from src.networks.subset9Modular36 import subnet9mod36

### Saving figures setup ###
### if we want to not save, just set save = False ###
figureDir = "../results/figures/"
savefig = True
###########################

### Saving results setup ###
### if we want to not save, just set savemat = False ###
dataDir = "../results/data/"
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
    return (global_coupling, data, time)

# running a parameter sweep of the global coupling parameter
gc_range = np.arange(0.0, 5.1, .50)
data = []
for gc in gc_range:
    data.append((run_sim(np.array([gc]))))

#plot connectivity
f1 = plt.figure()
utils.pyutils.connMatrixPlotter.connMatrixPlotter(conn)
plt.show()
#
# f2 = plt.figure(figsize=(21, 8))
# gs = f2.add_gridspec(1, 2)
# ax1 = f2.add_subplot(gs[0, 0])
# ax1.set_title('subset of 6')
# plt.plot(time, data[:, 4:])
# ax2 = f2.add_subplot(gs[0, 1])
# ax2.set_title('subset of 3')
# plt.plot(time,data[:,:3])
# plt.show()


# if savefig:
#     plt.savefig(figureDir + "figure{}".format() + ".png", dpi=600, bbox_inches='tight')
#
# if resultsmat:
#     sio.savemat(dataDir + "results{}".format() + ".mat", {'data':data})


