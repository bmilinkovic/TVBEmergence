import csv
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import scipy.io as sio
import scipy.stats as stat
import zipfile
from tvb.simulator.plot.tools import *
from tvb.simulator.lab import *
from tvb.simulator.models.wong_wang_exc_inh import ReducedWongWangExcInh

import os


data_dir = os.path.abspath("TVB_input")
zip_suffix = "_TVB"

def load_connectivity(input_name):
    zip_file_name = input_name + zip_suffix + ".zip"
    dir_name = input_name + zip_suffix
    zip_path = os.path.join(data_dir, input_name, zip_file_name)
    dir_path = os.path.join(data_dir, input_name, dir_name)
    #load connectivity data
    conn = connectivity.Connectivity.from_file(zip_path)
    # configure, to compute derived data.
    conn.configure()

    # check weight matrix from .zip is corresponding to structural connectivity matrix from matlab.
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dir_path)
    weight_txt = np.loadtxt(fname = dir_path + "/weights.txt")
    # load structural connectivity matrix from matlab file
    SC_path = data_dir + "/" + input_name + "/" + "/SCthrAn.mat"
    x = sio.loadmat(SC_path)
    assert np.allclose(x['SCthrAn'], weight_txt), "Weights matrix in weights.txt should be the same as SCthrAn.mat"
    return conn

# now since we have our function for loading the connectivity lets load in the data.

input_name = "CON02T2"
conn = load_connectivity(input_name)
conn.configure()

# plot the structural connectivity info: a 2d plot for visualising connectivity.weights matrix

plot_connectivity(connectivity=conn, plot_tracts=False)
plt.xlabel("regions")
plt.ylabel("regions")
connectivity_title = "Structural connectiivty for subject " + input_name
plt.title(connectivity_title, fontweight="bold",fontsize="12", y = 1.05)
plt.show()

# load empirical functional connectivity matrix from FC.mat

fc_file_name = os.path.join(data_dir, input_name, "FC.mat")
fc_cc_name = "FC_cc_DK68"
em_fc_matrix = sio.loadmat(fc_file_name)[fc_cc_name]

# indexes of all the true values above the diagonal.
uidx = np.triu_indices(68,1)
#Fisher-Z transform the correlations, important for standardisation
em_fc_z = np.arctanh(em_fc_matrix)
#Get the upper triangle since it is symmetric along the diagonal
em_fc = em_fc_z[uidx]

# set up region model
rww = ReducedWongWangExcInh()

# set up simulator

sim = simulator.Simulator(model=rww,
                          connectivity=conn,
                          coupling=coupling.Linear(),
                          integrator=integrators.HeunStochastic(dt=1, noise=noise.Additive(nsig=np.array([1e-5]))),
                          monitors=(monitors.TemporalAverage(period=2100.0),
                                    monitors.Bold(period=2100),
                                    monitors.ProgressLogger(period=1e5)
                                    ),
                                    simulation_length = 200000
).configure()

def compute_corr(time_line, data_result, sim):
    input_shape = data_result.shape
    result_shape = (input_shape[2], input_shape[2], input_shape[1], input_shape[3])
    sample_period = sim.monitors[1].period
    t_start = sample_period
    t_end = time_line[-1]
    t_lo = int((1. / sample_period) * (t_start - sample_period))
    t_hi = int((1. / sample_period) * (t_end - sample_period))
    t_lo = max(t_lo,0)
    t_hi = max(t_hi, input_shape[0])
    FC = np.zeros(result_shape)
    for mode in range(result_shape[3]):
        for var in range(result_shape[2]):
            current_slice =  tuple([slice(t_lo, t_hi +1), slice(var, var + 1), slice(input_shape[2]), slice(mode, mode + 1)])
            data = data_result[current_slice].squeeze()
            FC[:,:,var, mode] = np.corrcoef(data.T)
    return FC

def run_sim(global_coupling):
    sim.coupling.a = global_coupling
    print("Starting simulation... GC = " + str(global_coupling))
    (tavg_time, tavg_data), (bold_time, bold_data), _ = sim.run()

    print("Starting corrcoef")
    FC = compute_corr(bold_time, bold_data, sim)
    FC = FC[:,:,0,0]

    #Fisher-Z score that, important for standardisation
    sim_fc = np.arctanh(FC)[uidx]

    # calculate the link-wise pearson correlation between individuals upper triangular part of the simulated and empirical
    #functional connectivity matrix
    print("starting pearsonr")
    try:
        pearson_corr, _ = stat.pearsonr(sim_fc, em_fc)
    except ValueError:
        print("Simulation of Corr end in Nan..")
        pearson_corr = None

    return (global_coupling, pearson_corr)


# defining the global coupling range to explore in simulation
# in the original study a range from 0.01 to 3 with steps of 0.015 was explored
# NOTE: too many steps will take very long time when running the script on a local computer
# adjust the range of G, or the step size to reduce simulation time
gc_range = np.arange(0.01, 3, 0.29)
results = []
for gc in gc_range:
    results.append((gc, run_sim(np.array([gc]))))

# n_cores = 8
# p = mp.Pool(processes=n_cores)
# results = p.map(run_sim(np.array([gc]), gc_range)
# p.close()

# plotting time
g = []
pcorr = []
for result in results:
    g.append(result[0])
    pcorr.append(result[1][1])
pyplot.xlabel('G')
pyplot.ylabel('Corr')
pyplot.title('G vs Correlation')
plt.plot(g, pcorr)
plt.show()

