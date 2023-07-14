import numpy as np
import scipy as sc
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


from plots import ssdigraphs



data_dir = 'results/ssdiDataMATLAB'
files_dir = '/SJ3D_5node_withlink_ps_gc-noise/ssdiData/'

data_list = []
macrosize_list = []
for filename in os.listdir(data_dir + files_dir):
    if 'dynamical_dependence' in filename:
        f = sio.loadmat(os.path.join(data_dir + files_dir, filename))
        data = f['dynamical_independence_matrix']
        data_list.append(data)
        macrosize = int(filename.split('MACRO')[-2][-1])
        macrosize_list.append(macrosize)

data_array = np.dstack(data_list)
df = pd.DataFrame(data_array.reshape(-1, data_array.shape[2]), columns=[f"file_{i}" for i in range(data_array.shape[2])])
macrosize_array = np.array(macrosize_list)


# # Plotting the dynamical dependence values

ssdigraphs.plot_dd(data_array, [macrosize_array[i]])



# Load data

# figure_results = 'results/figures_for_publication'

# # path for optimisation and preoptimisation histories directories

# optimisation_dir = '/Users/borjanmilinkovic/Documents/gitdir/ssdi/networks/models/'

# # plotting preoptimisation and optimisation histories
# preopt = sio.loadmat(os.path.join(optimisation_dir, 'opthistp_mdim_2.mat'))
# preopthist = preopt['ohistp']
# gopt = sio.loadmat(os.path.join(optimisation_dir, 'optdistp_mdim_2.mat'))
# goptp = gopt['goptp']


# # for plotting optimisation and preoptimisation histories and distances

# fig = ssdigraphs.plot_optp(preopthist, gopt)
# fig.savefig(os.path.join(figure_results, "preopthistdist_FIG.eps"), dpi=300)
# fig.savefig(os.path.join(figure_results, "preopthistdist_FIG.png"), dpi=300)
# plt.show()




# plotting gc matrix and graph

#load in data
# eweight = sio.loadmat(os.path.join(optimisation_dir, 'sim_model_0339_15_06_2023.mat'))      # load in edge weights
# eweight = eweight['eweight']    # extract edge weights

# fig1 = ssdigraphs.plot_gc(eweight)
# plt.savefig(os.path.join(figure_results, "gcgraph.eps"), dpi=300)
# plt.savefig(os.path.join(figure_results, "gcgraph.png"), dpi=300)
# plt.show()

# laoad in node weights and plot that

# nweight = sio.loadmat(os.path.join(optimisation_dir, 'nweight_mdim_2.mat'))      # load in node weights
# nweight = nweight['nweight']    # extract node weights

# fig2 = ssdigraphs.plot_nweights(eweight, nweight, 2, 72)
# plt.savefig(os.path.join(figure_results, "macro_smear.eps"), dpi=300)
# plt.savefig(os.path.join(figure_results, "macro_smear.png"), dpi=300)
# plt.show()


# For plotting DD values

# for filename in os.listdir(data_dir + file_dir):
#     if filename.__contains__('dynamical_dependence'):
#         f = os.path.join(data_dir + file_dir + filename)
#         file = sio.loadmat(f)
#         data = file['dynamical_independence_matrix']
#         data = pd.DataFrame(data)
#         fig = ssdigraphs.plot_dd_simulations(data, filename[16])
#         fig.savefig(os.path.join(figure_results, filename[:-4] + "_FIG.eps"), dpi=300)
#         # fig.savefig(os.path.join(figure_results, filename[:-4] + "_FIG.png"), dpi=300)


# For plotting node weights

# for filename in os.listdir(data_dir + file_dir):
#     if filename.__contains__('nodeWeights'):
#         f = os.path.join(data_dir + file_dir + filename)
#         file = sio.loadmat(f)
#         data = file['maximalNodeWeights']
#         fig = ssdigraphs.plot_nodeweights_individual(data, filename[9])
#         fig.savefig(os.path.join(figure_results, filename[:-4] + "_FIG.svg"), dpi=300)
#         fig.savefig(os.path.join(figure_results, filename[:-4] + "_FIG.png"), dpi=300)



