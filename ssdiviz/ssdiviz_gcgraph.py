'''This is a module that will hopefully be able to perform all visualisation for ssdi code
is generated in matlab.'''

import networkx as nx
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns



#Load in (G-)causal graph data and extract edge weights.
matlabDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/data/'
resultsDir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/figures/'
eweights = sio.loadmat(matlabDir + 'eweights')
eweights = list(eweights.values())[3]


# function 1
G = nx.from_numpy_array(eweights, parallel_edges=True, create_using=nx.MultiDiGraph)
G.remove_edges_from(list(nx.selfloop_edges(G))) # remove self edges
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items()) #Extracts the edges and their corresponding weights into different tuples

### PLOT TIME

plt.figure(figsize=[20,20])
pos = nx.spring_layout(G, seed=7)  #Set up a "Spring Layout": Nodes repel from each other the stronger the weight, the closer the nodes
nx.draw_networkx_nodes(G, pos, node_size = 700, linewidths = 1.0, edgecolors='black')  #Draw edge nodes first, passing the _pos_ variable and node_size as argumets
nx.draw_networkx_edges(G, pos, arrows= True, arrowstyle="->", arrowsize=10.0, edgelist=edges,edge_color=weights,node_size = 700, width=3.0, connectionstyle='arc3,rad=0.03', edge_cmap=plt.cm.Blues) #Draw edges using the "edges" as the edge list ad weights as the edge colors, setting a colour map from matplotlib.pyplot
nx.draw_networkx_labels(G, pos, font_size=16, font_family="helvetica") #Draw node LABELS
edge_labels = dict([((u,v,), f"{d['weight']:.2f}") for u,v,d in G.edges(data=True)]) #Draw edge LABELS: firstly we set the edge_labels to 1 decimal place
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5, alpha=0.9, label_pos=0.33)
plt.tight_layout()
plt.savefig(resultsDir + "test10.svg")
plt.show()

# function 2
inter_optima_dist = sio.loadmat(matlabDir + 'inter-optima-dist')
inter_optima_dist = inter_optima_dist['goptp']
sns.heatmap(np.flipud(inter_optima_dist))
plt.show()






