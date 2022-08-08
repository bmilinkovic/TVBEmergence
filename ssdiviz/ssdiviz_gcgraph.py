'''This is a module that will hopefully be able to perform all visualisation for ssdi code
is generated in matlab.'''

import networkx as nx
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import matplotlib as mpl



#Load in (G-)causal graph data and extract edge weights.
matlabDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/data/'
resultsDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/figures/'
eweights = sio.loadmat(matlabDir + 'eweights')
eweights = list(eweights.values())[3]


# function 1: micro g-causal graph
G = nx.from_numpy_array(eweights, parallel_edges=True, create_using=nx.MultiDiGraph)
G.remove_edges_from(list(nx.selfloop_edges(G))) # remove self edges
edges, weights = zip(*nx.get_edge_attributes(G,'weight').items()) #Extracts the edges and their corresponding weights into different tuples

macro_gcgraph = sio.loadmat(matlabDir + 'macro-gcgraph')
macro_gcgraph = macro_gcgraph['nweight']

### PLOT TIME

# function 2: inter optima distances
inter_optima_dist = sio.loadmat(matlabDir + 'inter-optima-dist')
inter_optima_dist = inter_optima_dist['goptp']
sns.heatmap(np.flipud(inter_optima_dist), vmin=0.0, vmax = 1.0, center = 0.5, cmap = "Blues")
plt.show()


# function for micro and macro graphs.


fig = plt.figure(figsize=(55,15))
gs = GridSpec(nrows=1, ncols=3)


ax0 = fig.add_subplot(gs[0,0])
sns.heatmap(eweights, cmap=mpl.cm.bone_r, cbar_kws={'label': 'GC-matrix'}, center=1.2, linewidths=.6, annot=True)

ax1 = fig.add_subplot(gs[0,1])
ax1.set_title("G-Causal Graph of 9-node MVAR(3) model")
pos = nx.spring_layout(G, seed=7)  #Set up a "Spring Layout": Nodes repel from each other the stronger the weight, the closer the nodes
nx.draw_networkx_nodes(G, pos, node_size = 1800, node_color='lightgray', linewidths = 1.0, edgecolors='black')  #Draw edge nodes first, passing the _pos_ variable and node_size as argumets
nx.draw_networkx_edges(G, pos, arrows= True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights,node_size = 1800, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r) #Draw edges using the "edges" as the edge list ad weights as the edge colors, setting a colour map from matplotlib.pyplot
nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica") #Draw node LABELS
edge_labels = dict([((u,v,), f"{d['weight']:.2f}") for u,v,d in G.edges(data=True)]) #Draw edge LABELS: firstly we set the edge_labels to 1 decimal place
#nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5, alpha=0.9, label_pos=0.3, verticalalignment='baseline')

ax2 = fig.add_subplot(gs[0,2])
ax2.set_title("G-Causal Graph of 9-node MVAR(3) model with projected Macro Variable of size = 3")
pos = nx.spring_layout(G, seed=7)  #Set up a "Spring Layout": Nodes repel from each other the stronger the weight, the closer the nodes
nx.draw_networkx_nodes(G, pos, node_size = 1800, node_color=macro_gcgraph[:,0], cmap=plt.cm.Blues, linewidths = 1.0, edgecolors='black')  #Draw edge nodes first, passing the _pos_ variable and node_size as argumets
nx.draw_networkx_edges(G, pos, arrows= True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights, node_size = 1800, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r) #Draw edges using the "edges" as the edge list ad weights as the edge colors, setting a colour map from matplotlib.pyplot
nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica") #Draw node LABELS
edge_labels = dict([((u,v,), f"{d['weight']:.1f}") for u,v,d in G.edges(data=True)]) #Draw edge LABELS: firstly we set the edge_labels to 1 decimal place
#nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5, alpha=0.9, label_pos=0.3, verticalalignment="bottom")

fig.tight_layout()
fig.savefig(resultsDir + "gc-causal-graph-and-micro-02.svg")
fig.show()