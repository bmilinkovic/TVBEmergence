import os
import time

import numpy as np
import scipy.io as sio

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import networkx as nx

import pandas as pd


# Below: directories for saved files of MATLAB SSDI analysis.
# They Include:
#               1. Minimal Dynamical Dependence (Maximal Dynamical Independence)
#               Across all simulations in a (1, N) vector, where N = # of sims
#               2. Edge Weights (Pairwise Granger-causal Graph) for each simulation
#               3. Node Weights: weights that represent the distance from each node
#               to the macroscopic variable, identifying which nodes fall into the macro
resultsDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/ssdiDataMATLAB'

# Loading: Osc2dGen data
# ssdiOscDD = os.path.join(resultsDir, "osc2d_3node_nodelay_a-0.7_ps_gc-noise-larger/ssdiData/dynamical_dependence_parametersweep_noise_gc-noiselarger.mat")
# ssdiOscEW = os.path.join(resultsDir, "osc2d_3node_nodelay_a-0.7_ps_gc-noise-larger/ssdiData/edgeWeights_parametersweep_noise_gc-noiselarger.mat")
# ssdiOscNW = os.path.join(resultsDir, "osc2d_3node_nodelay_a-0.7_ps_gc-noise-larger/ssdiData/nodeWeights_parametersweep_noise_gc-noiselarger.mat")

# Loading: SJ3D data
ssdiSJ3DDD = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/ssdiDataMATLAB/SJ3D_NOCONN_5node_nodelay_ps_gc-noise/ssdiData/SJ3D_NOCONN_AIC_3MACRO_dynamical_dependence_parametersweep_noise_gc.mat'
ssdiSJ3DNW = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/ssdiDataMATLAB/SJ3D_NOCONN_5node_nodelay_ps_gc-noise/ssdiData/SJ3D_NOCONN_AIC_3MACRO_nodeWeights_parametersweep_noise_gc.mat'
ssdiSJ3DEW = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/ssdiDataMATLAB/SJ3D_NOCONN_5node_nodelay_ps_gc-noise/ssdiData/SJ3D_NOCONN_AIC_3MACRO_edgeWeights_parametersweep_noise_gc.mat'

# Below: save directory for figures generated here
figuresDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/'
ssdiFigures = os.path.join(figuresDir, "ssdiFiguresPython/")

if not os.path.exists(ssdiFigures):
    os.makedirs(ssdiFigures)

# Below: Loading in Edge weights
edgeWeights = sio.loadmat(ssdiSJ3DEW)                           # loads in edge weights as estimated by MVGC
edgeWeights = edgeWeights['edgeWeights']


# edgeFrame = pd.DataFrame(edgeWeights)
# subset = edgeFrame.loc[:, 10:14]
# subset.columns = ['rTCI', 'rA2', 'lM1', 'lTCC', 'rIP']
# subset.index = ['rTCI', 'rA2', 'lM1', 'lTCC', 'rIP']


# Below: Loading in Node weights
nodeWeights = sio.loadmat(ssdiSJ3DNW)                           # loads in the subspace distances between macro and micro
nodeWeights = nodeWeights['maximalNodeWeights']              # selects the nweights dict. key.

# Below: Loading in Dynamical Dependence vector across simulations
ddData = sio.loadmat(ssdiSJ3DDD)
ddDataMatrix = ddData['dynamical_independence_matrix']
ddDataFrame = pd.DataFrame(ddDataMatrix)


# Creating GC string vector for axis plotting
global_coupling_log = 10**np.r_[-2:-0.5:15j]
global_coupling_log_round = [round(float(i), 3) for i in global_coupling_log]
global_coupling_log_str = [str(x) for x in global_coupling_log_round]

# Creating Noise string vector for axis plotting
noise_log = 10**np.r_[-3:-0.002:15j]
noise_log_round = [round(float(i), 3) for i in noise_log]
noise_log_str = [str(x) for x in noise_log_round]

# Setting them as column and index names in DDDataFrame
ddDataFrame.columns = [noise_log_str]
ddDataFrame.index = [global_coupling_log_str]

# below is code too loop over all edge weights from the different parameter runs and can be used in below graphing loop.
edgeIndex = np.arange(0, 1126, 5)  # set the indices upon which to loop over. this requires you to know the step size
# edgeWeightSubset = []               # is the size of the multivariate system of the number of nodes.
# edgeWeightsSubset = edgeWeights[:, edgeIndex[i]:edgeIndex[i + 1]]
for i in range(225):            # the range is over the amount of simulations I have performed in my parameter sweep.
    subset = pd.DataFrame(edgeWeights[:, edgeIndex[i]:edgeIndex[i+1]])
    subset.columns = ['rTCI', 'rA2', 'lM1', 'lTCC', 'rIP']
    subset.index = ['rTCI', 'rA2', 'lM1', 'lTCC', 'rIP']
    G = nx.from_pandas_adjacency(subset, create_using=nx.MultiDiGraph)
    G.remove_edges_from(list(nx.selfloop_edges(G)))                                             # remove self edges
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())                          # Extracts the edges and their corresponding weights into different tuples

    # PLOTTING THE PWCGC MATRIX, GRAPH AND MACRO PROJECTION ON GRAPH.
    fig = plt.figure(figsize=(24, 8))
    gs = GridSpec(nrows=1, ncols=3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title("Pairwise Granger-causality Matrix", fontsize=20, fontweight='bold', pad=16)
    sns.heatmap(subset, cmap=mpl.cm.bone_r, center=0.5, linewidths=.6, annot=True)
    ax0.invert_yaxis()

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title("GC-graph of an Uncoupled {0}-node SJ3D model".format(int(len(subset))), fontsize=20, fontweight='bold', pad=16)
    pos = nx.spring_layout(G, seed=7)
    nx.draw_networkx_nodes(G, pos, node_size=1600, node_color='lightgray', linewidths=1.0, edgecolors='black')
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights,node_size=1600, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r)
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica")
    edge_labels = dict([((u, v,), f"{d['weight']:.2f}") for u, v, d in G.edges(data=True)])


    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title("3-Macro on GC-graph of Uncoupled {0}-node SJ3D model".format(int(len(subset))), fontsize=20, fontweight='bold', pad=16)
    pos = nx.spring_layout(G, seed=7)
    nx.draw_networkx_nodes(G, pos, node_size=1600, node_color=nodeWeights[:, i], cmap=plt.cm.Blues, linewidths=1.0, edgecolors='black')
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights, node_size=1600, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r)
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica")
    edge_labels = dict([((u, v,), f"{d['weight']:.1f}") for u, v, d in G.edges(data=True)])


    fig.tight_layout()
    #fig.savefig(ssdiFigures + "SJ3D_AIC_5node_nodelay_ps_gc-noise-nodeweights-{0}".format(int(134)))
    fig.savefig(ssdiFigures + "SJ3D_NOCONN_AIC_3macro_GCMACROPLOT/SJ3D_AIC_5node_nodelay_ps_gc-noise-nodeweights-{0}".format(int(i)))
    #fig.show()
    fig.clf()

# node 1
node1 = nodeWeights[0,:].reshape(15,15)
node1Frame = pd.DataFrame(node1)
node1Frame.columns = [noise_log_str]
node1Frame.index = [global_coupling_log_str]

node2 = nodeWeights[1,:].reshape(15,15)
node2Frame = pd.DataFrame(node2)
node2Frame.columns = [noise_log_str]
node2Frame.index = [global_coupling_log_str]

node3 = nodeWeights[2,:].reshape(15,15)
node3Frame = pd.DataFrame(node3)
node3Frame.columns = [noise_log_str]
node3Frame.index = [global_coupling_log_str]

node4 = nodeWeights[3,:].reshape(15,15)
node4Frame = pd.DataFrame(node4)
node4Frame.columns = [noise_log_str]
node4Frame.index = [global_coupling_log_str]

node5 = nodeWeights[4,:].reshape(15,15)
node5Frame = pd.DataFrame(node5)
node5Frame.columns = [noise_log_str]
node5Frame.index = [global_coupling_log_str]


# Below: Plotting ddDataMatrix & nodeDataMatrix
# PLOTTING PARAMETER SWEEPS WITH DD VALUES OF EACH PARAMETER COMBINATION
fig = plt.figure(figsize=(50, 25))
gs = GridSpec(nrows=2, ncols=4)

ax0 = fig.add_subplot(gs[0, 0])
ax0.set_title('Dynamical Dependence change over Parameter Regimes', fontsize=20, fontweight='bold', pad=16)
ax0 = sns.heatmap(ddDataFrame, cmap='bone_r',)
ax0.set_xlabel('Noise', fontsize=16, fontweight='bold', labelpad=10)
ax0.set_ylabel('Global Coupling', fontsize=16, fontweight='bold', labelpad=10)
ax0.invert_yaxis()


ax1 = fig.add_subplot(gs[0, 1])
ax1.set_title('Weight of Node 1 across parameter selections', fontsize=20, fontweight='bold', pad=16)
ax1 = sns.heatmap(node1Frame, cmap='bone_r',)
ax1.set_xlabel('Noise', fontsize=16, fontweight='bold', labelpad=10)
ax1.set_ylabel('Global Coupling', fontsize=16, fontweight='bold', labelpad=10)
ax1.invert_yaxis()

ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('Node-weight by contribution to 3-MACRO for each Parameter Regime', fontsize=20, fontweight='bold', pad=16)
ax2 = sns.heatmap(nodeWeights, yticklabels=np.flip(['rTCI', 'rA2', 'lM1', 'lTCC', 'rIP']), cmap='bone_r')
ax2.set_xlabel('Parameter Sweep Run', fontsize=16, fontweight='bold', labelpad=10)
ax2.set_ylabel('Brain Region (Node)', fontsize=16, fontweight='bold', labelpad=10)

ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title('Weight of Node 2 across parameter selections', fontsize=20, fontweight='bold', pad=16)
ax3 = sns.heatmap(node2Frame, cmap='bone_r',)
ax3.set_xlabel('Noise', fontsize=16, fontweight='bold', labelpad=10)
ax3.set_ylabel('Global Coupling', fontsize=16, fontweight='bold', labelpad=10)
ax3.invert_yaxis()

ax4 = fig.add_subplot(gs[0, 2])
ax4.set_title('Weight of Node 3 across parameter selections', fontsize=20, fontweight='bold', pad=16)
ax4 = sns.heatmap(node3Frame, cmap='bone_r',)
ax4.set_xlabel('Noise', fontsize=16, fontweight='bold', labelpad=10)
ax4.set_ylabel('Global Coupling', fontsize=16, fontweight='bold', labelpad=10)
ax4.invert_yaxis()

ax5 = fig.add_subplot(gs[1, 2])
ax5.set_title('Weight of Node 4 across parameter selections', fontsize=20, fontweight='bold', pad=16)
ax5 = sns.heatmap(node4Frame, cmap='bone_r',)
ax5.set_xlabel('Noise', fontsize=16, fontweight='bold', labelpad=10)
ax5.set_ylabel('Global Coupling', fontsize=16, fontweight='bold', labelpad=10)
ax5.invert_yaxis()

ax6 = fig.add_subplot(gs[0, 3])
ax6.set_title('Weight of Node 5 across parameter selections', fontsize=20, fontweight='bold', pad=16)
ax6 = sns.heatmap(node5Frame, cmap='bone_r',)
ax6.set_xlabel('Noise', fontsize=16, fontweight='bold', labelpad=10)
ax6.set_ylabel('Global Coupling', fontsize=16, fontweight='bold', labelpad=10)
ax6.invert_yaxis()

fig.tight_layout()
fig.savefig(ssdiFigures + "SJ3D_NOCONN_AIC_5node_3MACRO_nodelay_ps_gc-noise_parameterSweep_and_macroWeightingforALLNODES.svg")
fig.show()








