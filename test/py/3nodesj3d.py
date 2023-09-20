import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy as sc
import scipy.io as sio
from tvb.simulator.lab import *
from scipy.stats import zscore
import os


# setting up the directories

results_directory = './results/data/'

if not os.path.exists(results_directory):
    os.makedirs(results_directory)



conn = connectivity.Connectivity()
conn.centres_spherical(number_of_regions=3)
conn.weights = np.array([
                            [0, 2, 0], 
                            [1, 0, -2], 
                            [1, 0.5, 0]
])
conn.tract_lengths = np.array([
                                [0, 8.8, 0],
                                [8.8, 0, 0], 
                                [0, 0, 0]
])
conn.region_labels = np.array(['Cortex', 'Thalamus', 'Reticular T'])
# conn = connectivity.Connectivity(weights = np.array([[0,2,0], [1,0,-2], [1,0.5,0]]),
#                                  tract_lengths = np.array([[0,8.8,0],[8.8,0,0], [0,0,0]]),
#                                region_labels = np.array(['Cortex', 'Thalamus', 'Reticular T']))

conn.configure()


# Plotting
plot_matrix(conn.weights, connectivity=conn)
plt.savefig(results_directory + 'weights.png')
plt.show()

plot_matrix(conn.tract_lengths, connectivity=conn)
plt.savefig(results_directory + 'tract_lengths.png')
plt.show()

local_mod = models.ReducedSetHindmarshRose(K21 = np.array([0.5]), K12 = np.array([0.25]), mu= np.array([3.1]), sigma=np.array(0.5))
coupling = coupling.Linear(a=np.array([0.5]))
integ = integrators.HeunStochastic(dt=2**-6, noise=noise.Additive(nsig = np.array([0.44])))

mon_raw = monitors.Raw()
mon_subsample = monitors.SubSample(period=1)
mon = (mon_raw, mon_subsample)



sim = simulator.Simulator(model=local_mod,
                          connectivity = conn,
                          coupling = coupling,
                          integrator = integ,
                          monitors = mon)
sim.configure()

raw_data = []
raw_time = []
sub_data = []
sub_time = []

for raw, sub in sim(simulation_length=5000):
    if not raw is None:
        raw_time.append(raw[0])
        raw_data.append(raw[1])

    if not sub is None:
        sub_time.append(sub[0])
        sub_data.append(sub[1])


# a = np.arange(0, 10, 0.2)
#
# for i, a_i in enumerate(a):
#     results = runsim(np.array([a_i]))
#     np.save('../results/results_%03d.npy', results)

RAW = np.array(raw_data)
SUB = np.array(sub_data)

var1 = zscore(np.sum(SUB[1000:,0,0,:], axis=1))
var2 = zscore(np.sum(SUB[1000:,0,1,:], axis=1))
var3 = zscore(np.sum(SUB[1000:,0,2,:], axis=1))

mvar = np.stack((var1, var2, var3), axis = 0)

# plotting

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Z-scored time-series of mean-field activity')
ax1.plot(var1[0:5000], "tab:green")
ax1.set_title('node1')
ax2.plot(var2[0:5000], "tab:green")
ax2.set_title('node2')
ax3.plot(var3[0:5000], "tab:blue")
ax3.set_title('node3')
fig.savefig(results_directory + 'mean_field_activity.png')
plt.show()

# saving the data as a matlab file
filename = os.path.join(results_directory, "3nodeSJ3D_1.mat")
sio.savemat(filename, {'mvar': mvar})





