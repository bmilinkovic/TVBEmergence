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
ssdiDD = os.path.join(resultsDir, "osc2d_3node_nodelay_a-0.7_ps_gc-noise-larger/ssdiData/dynamical_dependence_parametersweep_noise_gc-noiselarger.mat")
ssdiEW = os.path.join(resultsDir, "osc2d_3node_nodelay_a-0.7_ps_gc-noise-larger/ssdiData/edgeWeights_parametersweep_noise_gc-noiselarger.mat")
ssdiNW = os.path.join(resultsDir, "osc2d_3node_nodelay_a-0.7_ps_gc-noise-larger/ssdiData/nodeWeights_parametersweep_noise_gc-noiselarger.mat")

# Below: save directory for figures generated here
figuresDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/'
ssdiFigures = os.path.join(figuresDir, "ssdiFiguresPython/")

if not os.path.exists(ssdiFigures):
    os.makedirs(ssdiFigures)

# Below: Loading in Edge weights
edgeWeights = sio.loadmat(ssdiEW)                           # loads in edge weights as estimated by MVGC
edgeWeights = edgeWeights['edgeWeights']

# Below: Loading in Node weights
nodeWeights = sio.loadmat(ssdiNW)                           # loads in the subspace distances between macro and micro
nodeWeights = nodeWeights['maximalNodeWeights']              # selects the nweights dict. key.

edgeFrame = pd.DataFrame(edgeWeights)
subset = edgeFrame.loc[:, 1:3]
subset.columns = ['IPCs', 'DLPFC', 'V1']
subset.index = ['IPCs', 'DLPFC', 'V1']

# Below: Loading in Dynamical Dependence vector across simulations
ddData = sio.loadmat(ssdiDD)
ddDataMatrix = ddData['dynamical_independence_matrix']



G = nx.from_pandas_adjacency(subset, create_using=nx.MultiDiGraph)
G.remove_edges_from(list(nx.selfloop_edges(G)))                                             # remove self edges
edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())                          # Extracts the edges and their corresponding weights into different tuples
# # Graphing distances between inter-optimal subspaces for given n-Macro
#
# interOptimaFile = 'put filename here'
# interOptimaDistance = sio.loadmat(ssdiDirData + interOptimaFile)
# interOptimaDistance = interOptimaDistance['goptp']
# sns.heatmap(np.flipud(interOptimaDistance), vmin=0.0, vmax=1.0, center=0.5, cmap="Blues")
# plt.show()
#
#
# Graphing 1. GC-matrix, 2. GC-graph, 3. GC-graph with projected macro-variable.

fig = plt.figure(figsize=(24, 8))
gs = GridSpec(nrows=1, ncols=3)

ax0 = fig.add_subplot(gs[0, 0])
ax0.set_title("Pairwise Granger-causality Matrix", fontsize = 10)
sns.heatmap(subset, cmap=mpl.cm.bone_r, center=0.5, linewidths=.6, annot=True)

ax1 = fig.add_subplot(gs[0, 1])
ax1.set_title("GC-graph of {3}-node Generic2DOscillator model")#.format(int(len())))
pos = nx.spring_layout(G, seed=7)
nx.draw_networkx_nodes(G, pos, node_size=1600, node_color='lightgray', linewidths=1.0, edgecolors='black')
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights,node_size=1600, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r)
nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica")
edge_labels = dict([((u, v,), f"{d['weight']:.2f}") for u, v, d in G.edges(data=True)])


ax2 = fig.add_subplot(gs[0, 2])
ax2.set_title("GC-graph of 3-node Gen2DOscillator model with projected Macro Variable of size = 2")
pos = nx.spring_layout(G, seed=7)
nx.draw_networkx_nodes(G, pos, node_size=1600, node_color=nodeWeights[:, 0], cmap=plt.cm.Blues, linewidths=1.0, edgecolors='black')
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights, node_size=1600, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r)
nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica")
edge_labels = dict([((u, v,), f"{d['weight']:.1f}") for u, v, d in G.edges(data=True)])


fig.tight_layout()
fig.savefig(ssdiFigures + "osc2d_3node_nodelay_a-0.7_ps_gc_noiselarger_microMacroGraph.svg")
fig.show()



# Below: Plotting ddDataMatrix & nodeDataMatrix
fig = plt.figure(figsize=(20,10))
gs = GridSpec(nrows=1, ncols=2)

ax0 = fig.add_subplot(gs[0,0])
ax0.set_title('Parameter Sweep across Global Coupling and Noise')
plt.xscale('log')
plt.yscale('log')
ax0 = sns.heatmap(ddDataMatrix, cmap='bone_r')
ax0.invert_yaxis()

ax1 = fig.add_subplot(gs[0,1])
ax1.set_title('Node weighting across every optimisation run')
sns.heatmap(nodeWeights, cmap='bone_r')

fig.tight_layout()
#fig.savefig(ssdiFigures + "osc2d_3node_nodelay_a-0.7_ps_gc_noiselarger_parameterSweep_and_macroWeighting.svg")
fig.show()
