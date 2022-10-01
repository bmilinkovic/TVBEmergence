import os
import time


import numpy as np
import scipy.io as sio

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import networkx as nx


# Directories for data structures saved in the matlab ssdi analysis and the directory for graphing the figures.

resultsDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/'
ssdiDirData = os.path.join(resultsDir, "ssdi/data/")
ssdiDirFigures = os.path.join(resultsDir, "ssdi/figures/")

if not os.path.exists(ssdiDirData):
    os.makedirs(ssdiDirData)

if not os.path.exists(ssdiDirFigures):
    os.makedirs(ssdiDirFigures)


edgeWeightsFile = 'put filename here'
edgeWeights = sio.loadmat(ssdiDirData + edgeWeightsFile)    # loads in edge weights as estimated by MVGC
edgeWeights = list(edgeWeights.values())[3]                 # creates a list object.

macroWeightsFile = 'put filename here'
macroWeights = sio.loadmat(ssdiDirData + 'macro-gcgraph')   # loads in the subspace distances between macro and micro
macroWeights = macroWeights['nweight']                      # selects the nweights dict. key.

G = nx.from_numpy_array(edgeWeights, parallel_edges=True, create_using=nx.MultiDiGraph)
G.remove_edges_from(list(nx.selfloop_edges(G)))                                             # remove self edges
edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())                          # Extracts the edges and their corresponding weights into different tuples

# Graphing distances between inter-optimal subspaces for given n-Macro

interOptimaFile = 'put filename here'
interOptimaDistance = sio.loadmat(ssdiDirData + interOptimaFile)
interOptimaDistance = interOptimaDistance['goptp']
sns.heatmap(np.flipud(interOptimaDistance), vmin=0.0, vmax=1.0, center=0.5, cmap="Blues")
plt.show()


# Graphing 1. GC-matrix, 2. GC-graph, 3. GC-graph with projected macro-variable.

fig = plt.figure(figsize=(24, 8))
gs = GridSpec(nrows=1, ncols=3)


ax0 = fig.add_subplot(gs[0, 0])
ax0.set_title("Pairwise Granger-causality Matrix", fontsize = 10)
sns.heatmap(eweights, cmap=mpl.cm.bone_r, center=0.5, linewidths=.6, annot=True)

ax1 = fig.add_subplot(gs[0, 1])
ax1.set_title("GC-graph of {3}-node MVAR(3) model"#.format(int(len())))
pos = nx.spring_layout(G, seed=7)
nx.draw_networkx_nodes(G, pos, node_size=1600, node_color='lightgray', linewidths=1.0, edgecolors='black')
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights,node_size=1600, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r)
nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica")
edge_labels = dict([((u, v,), f"{d['weight']:.2f}") for u, v, d in G.edges(data=True)])


ax2 = fig.add_subplot(gs[0, 2])
ax2.set_title("GC-graph of 9-node MVAR(3) model with projected Macro Variable of size = 3")
pos = nx.spring_layout(G, seed=7)
nx.draw_networkx_nodes(G, pos, node_size=1600, node_color=macro_gcgraph[:, 0], cmap=plt.cm.Blues, linewidths=1.0, edgecolors='black')
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights, node_size=1600, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r)
nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica")
edge_labels = dict([((u, v,), f"{d['weight']:.1f}") for u, v, d in G.edges(data=True)])


fig.tight_layout()
fig.savefig(resultsDir + "gc-causal-graph-and-micro-02.svg")
fig.show()
