import numpy as np
from tvb.simulator.lab import *

def runsim(a_i):
    conn = connectivity.Connectivity.from_file()
    mod = models.Generic2dOscillator(a=a_i)
    integ = integrators.HeunStochastic()
    mon_subsample = monitors.SubSample(period=1)
    sim = simulator.Simulator(connectivity = conn,
                              model=mod,
                              integrator = integ,
                              monitors = (mon_subsample,)
                              )
    sim.configure()
    results = []
    for (t, data), in sim(simulation_length=1000):
        results.append(data)
    return np.array(results)

a = np.arange(0.0, 4, 1.0)
results = []
for a_i in a:
    results.append(runsim(np.array([a_i])))
    np.save('../results/data/results_new.npy', results)
