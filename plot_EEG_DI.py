#%%
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

#%%
directory = "/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/AnesthesiaProjectEmergence/results/ssdiData/"
#data = scipy.io.loadmat(os.path.join(directory, 'H0010P_mdim_2_dynamical_dependence.mat'))

#%%
fig, ax = plt.subplots()

dynamical_dependence_data = []

# BEGIN: 7f9d8a9d5f8c
# Define the colors for each label
colors = {'W': 'red', 'K': 'green', 'S': 'blue', 'X': 'orange', 'P': 'purple'}

for filename in os.listdir(directory):
    if filename.endswith("mdim_2_dynamical_dependence.mat"):
        # Load the data file
        data = scipy.io.loadmat(os.path.join(directory, filename))

        # Append the data as column vectors
        dynamical_dependence_data.append(data['dopto'][0][:])

        # Get the first 6 characters of the file name
        label = filename[5:6]

        # Add the violin plot with the label and color
        ax.violinplot(data['dopto'][0][:], positions=[len(dynamical_dependence_data)], showmeans=False, showmedians=False, showextrema=False, facecolor=colors[label])
        ax.text(len(dynamical_dependence_data), np.min(data['dopto'][0][:]), label, ha='center', va='top')

# Plot the data
ax.violinplot(dynamical_dependence_data)

ax.set_ylabel("Dynamical Dependence")
ax.set_title("Dynamical Dependence for 2-macro")

# Save the figure to a file
fig.savefig(os.path.join('./results/dd_EEG/', 'dd_mdim_2.png'), dpi=300, bbox_inches='tight')

# Show the figure
plt.show()
# END: 7f9d8a9d5f8c


#fig.show()

#%%
dynamical_dependence_data = []

for filename in os.listdir(directory):
    if filename.endswith(".mat") and "dynamical_dependence" in filename:
        # Load the data file
        data = scipy.io.loadmat(os.path.join(directory, filename))

                # Append the data as column vectors
        dynamical_dependence_data.append(data['dopto'][0][:])


#%% Plotting the node weights of the whole brain network


directory = "/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/AnesthesiaProjectEmergence/results/ssdiData/"
filename_nodeweights = "H0905X_mdim_18_node_weights.mat"™lo0990990

filename_edgeweights = "/Volumes/dataSets/restEEGHealthySubjects/restEEGHealthySubjects/AnesthesiaProjectEmergence/results/pwcgc_matrix/pwcgc_matrix_H0905X_source_time_series_34-of-34.mat"

nweight = scipy.io.loadmat(os.path.join(directory, filename_nodeweights)) # load in node weights
nweight = nweight['node_weights']    # extract node weights

eweight = scipy.io.loadmat(filename_edgeweights) # load in edge weights
eweight = eweight['edgeWeightsMatrix']    # extract edge weights

#%%
def plot_nweights_wholebrain(eweights, nweights, macrosize, opt_number):
        subset = pd.DataFrame(eweights)
        # subset.columns = ['0', '1', '2']
        # subset.index = ['0','1','2']
        G = nx.from_pandas_adjacency(subset, create_using=nx.MultiDiGraph)
        G.remove_edges_from(list(nx.selfloop_edges(G)))                    # remove self edges
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items()) # Extracts the edges and their corresponding weights into different tuples

        # PLOTTING THE PWCGC MATRIX, GRAPH AND MACRO PROJECTION ON GRAPH.
        fig = plt.figure(figsize=(16, 16))
        gs = GridSpec(nrows=1, ncols=1)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_title("{0}-Macro on GC-graph of coupled {1}-node model".format(int(macrosize), int(len(subset))), fontsize=12, fontweight='bold', pad=16)
        pos = nx.spring_layout(G, seed=7)
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color=nweights[:,opt_number], cmap=plt.cm.Blues, linewidths=1.0, edgecolors='black') # nweights[:,0] will plot the optimal projection of the first macro variable on the graph.
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights, node_size=800, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r)
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica")
        edge_labels = dict([((u, v,), f"{d['weight']:.1f}") for u, v, d in G.edges(data=True)])

        return fig

# %%
figure = plot_nweights_wholebrain(eweight, nweight, 18, 0)
plt.show()
# %%
