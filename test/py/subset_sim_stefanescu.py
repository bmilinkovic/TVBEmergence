from tvb.simulator.lab import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import utils.pyutils.connMatrixPlotter
from src.networks.subset9Modular36 import subnet9mod36



conn = subnet9mod36()
local_mod = models.ReducedSetHindmarshRose(K21 = np.array([0.5]), K12 = np.array([0.25]), mu= np.array([3.1]), sigma=np.array(0.5))
coupling = coupling.Linear(a=np.array([0.5]))
integ = integrators.HeunStochastic(dt=0.05, noise=noise.Additive(nsig = np.array([0.004])))

# defining a function to run a simulation: this will be useful for parameter sweeps


def run_sim(global_coupling):
    sim.coupling.a = global_coupling
    print("Starting SJ3D simulation with coupling factor = " + str(global_coupling))
    (sub_time, sub_data) = sim.run()


monitors = monitors.SubSample(period=1)
monitors.configure()

# setting simulation
simulation = simulator.Simulator(connectivity=conn,
                                 coupling=coupling,
                                 integrator=integ,
                                 model=local_mod,
                                 monitors=[monitors],
                                 simulation_length=5000)
simulation.configure()

results = simulation.run()
time = results[0][0]
data = results[0][1].squeeze()

#plot connectivity
f1 = plt.figure()
utils.pyutils.connMatrixPlotter.connMatrixPlotter(connectivity)
plt.show()

f2 = plt.figure(figsize=(21, 8))
gs = f2.add_gridspec(1, 2)
ax1 = f2.add_subplot(gs[0, 0])
ax1.set_title('subset of 6')
plt.plot(time, data[:, 4:])
ax2 = f2.add_subplot(gs[0, 1])
ax2.set_title('subset of 3')
plt.plot(time,data[:,:3])
plt.show()

# saving file to .mat file for use in G-causality.
sio.savemat('../results/data/simulationData6.mat', {'data': data})



