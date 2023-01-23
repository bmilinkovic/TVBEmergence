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
ssdiSJ3DDD = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/ssdiDataMATLAB/SJ3D_3node_withlink_ps_gc-noise/ssdiData/dynamical_dependence_parametersweep_noise_gc.mat'
ssdiSJ3DEW = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/ssdiDataMATLAB/SJ3D_3node_withlink_ps_gc-noise/ssdiData/edgeWeights_parametersweep_noise_gc.mat'
ssdiSJ3DNW = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/ssdiDataMATLAB/SJ3D_3node_withlink_ps_gc-noise/ssdiData/nodeWeights_parametersweep_noise_gc.mat'

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
global_coupling_log = 10**np.r_[-2:-0.7:20j]
global_coupling_log_round = [round(float(i), 3) for i in global_coupling_log]
global_coupling_log_str = [str(x) for x in global_coupling_log_round]

# Creating Noise string vector for axis plotting
noise_log = 10**np.r_[-6:-1:20j]
noise_log_round = [round(float(i), 3) for i in noise_log]
noise_log_str = [str(x) for x in noise_log_round]

# Setting them as column and index names in DDDataFrame
ddDataFrame.columns = [noise_log_str]
ddDataFrame.index = [global_coupling_log_str]

# below is code too loop over all edge weights from the different parameter runs and can be used in below graphing loop.
edgeIndex = np.arange(0, 2001, 5)  # set the indices upon which to loop over. this requires you to know the step size
# edgeWeightSubset = []               # is the size of the multivariate system of the number of nodes.
# edgeWeightsSubset = edgeWeights[:, edgeIndex[i]:edgeIndex[i + 1]]

def plot_macro_gc(edge_weights, node_weights, trials):

    for i in range(trials):            # the range is over the amount of simulations performed.
        subset = pd.DataFrame(edge_weights[:, edge_weights[i]:edge_weights[i+1]])
        subset.columns = ['rTCI', 'rA2', 'lM1', 'lTCC', 'rIP']
        subset.index = ['rTCI', 'rA2', 'lM1', 'lTCC', 'rIP']
        G = nx.from_pandas_adjacency(subset, create_using=nx.MultiDiGraph)
        G.remove_edges_from(list(nx.selfloop_edges(G)))                    # remove self edges
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items()) # Extracts the edges and their corresponding weights into different tuples

        # PLOTTING THE PWCGC MATRIX, GRAPH AND MACRO PROJECTION ON GRAPH.
        fig = plt.figure(figsize=(24, 8))
        gs = GridSpec(nrows=1, ncols=3)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_title("Pairwise Granger-causality Matrix", fontsize=30, fontweight='bold', pad=16)
        sns.heatmap(subset, cmap=mpl.cm.bone_r, center=0.5, linewidths=.6, annot=True)
        ax0.invert_yaxis()

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.set_title("GC-graph of an Uncoupled {0}-node SJ3D model".format(int(len(subset))), fontsize=30, fontweight='bold', pad=16)
        pos = nx.spring_layout(G, seed=7)
        nx.draw_networkx_nodes(G, pos, node_size=1600, node_color='lightgray', linewidths=1.0, edgecolors='black')
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights,node_size=1600, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r)
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica")
        edge_labels = dict([((u, v,), f"{d['weight']:.2f}") for u, v, d in G.edges(data=True)])


        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_title("3-Macro on GC-graph of Uncoupled {0}-node SJ3D model".format(int(len(subset))), fontsize=30, fontweight='bold', pad=16)
        pos = nx.spring_layout(G, seed=7)
        nx.draw_networkx_nodes(G, pos, node_size=1600, node_color=node_weights[:, i], cmap=plt.cm.Blues, linewidths=1.0, edgecolors='black')
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights, node_size=1600, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r)
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica")
        edge_labels = dict([((u, v,), f"{d['weight']:.1f}") for u, v, d in G.edges(data=True)])


        fig.tight_layout()
        #fig.savefig(ssdiFigures + "SJ3D_AIC_5node_nodelay_ps_gc-noise-nodeweights-{0}".format(int(134)))
        fig.savefig(ssdiFigures + "SJ3D_3node_withlink_2macro_GCMACROPLOT/SJ3D_3node_2MACRO_withlink_ps_gc-noise-nodeweights-{0}".format(int(i)))
        #fig.show()
        fig.clf()


# Node structures.
node1 = pd.DataFrame(nodeWeights[0, :].reshape(20, 20))
node1.columns = [noise_log_str]
node1.index = [global_coupling_log_str]

node2 = pd.DataFrame(nodeWeights[1, :].reshape(20, 20))
node2.columns = [noise_log_str]
node2.index = [global_coupling_log_str]

node3 = pd.DataFrame(nodeWeights[2, :].reshape(20, 20))
node3.columns = [noise_log_str]
node3.index = [global_coupling_log_str]

# node4 = pd.DataFrame(nodeWeights[3, :].reshape(20, 20))
# node4.columns = [noise_log_str]
# node4.index = [global_coupling_log_str]
#
# node5 = pd.DataFrame(nodeWeights[4, :].reshape(20, 20))
# node5.columns = [noise_log_str]
# node5.index = [global_coupling_log_str]





# Below: Plotting ddDataMatrix & nodeDataMatrix
# PLOTTING PARAMETER SWEEPS WITH DD VALUES OF EACH PARAMETER COMBINATION

def plot_dd_simulations(dynamical_dependence_values):

    # intialise variables:
    edge_cmap = sns.color_palette("YlOrBr", as_cmap=True)

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(nrows=1, ncols=1)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title('Dynamical Dependence change over Parameter Regimes', fontsize=30, fontweight='bold', pad=16)
    ax0 = sns.heatmap(dynamical_dependence_values, cmap=edge_cmap)
    ax0.set_xlabel('Noise', fontsize=22, fontweight='bold', labelpad=10)
    ax0.set_ylabel('Global Coupling', fontsize=22, fontweight='bold', labelpad=10)
    ax0.invert_yaxis()
    fig.savefig(os.path.join(ssdiFigures, "SJ3D_3node_withlink_2MACRO_ps_gc-noise_DDparameterSweep.svg"))


def plot_nodeweights_all(node_weights, subset):

    # initialise values:
    region_labels = subset.region_labels
    n_nodes = np.size(region_labels)

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(nrows=1, ncols=1)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title('Node-weight by contribution to 2-MACRO for each Parameter Regime', fontsize=34, fontweight='bold', pad=16)
    ax0 = sns.heatmap(node_weights, yticklabels=np.flip(region_labels), cmap='bone_r')
    ax0.set_xlabel('Parameter Sweep Run', fontsize=28, fontweight='bold', labelpad=10)
    ax0.set_ylabel('Brain Region (Node)', fontsize=28, fontweight='bold', labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=18, fontweight='bold', rotation=90)
    fig.savefig(os.path.join(ssdiFigures, "SJ3D_withlink_3node_2MACRO_ps_gc-noise_parameterSweep_NodeRaster.svg"))




def plot_nodeweights_individual(node1, node2, node3):

    fig = plt.figure(figsize=(45, 25))
    gs = GridSpec(nrows=2, ncols=2)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title('Weight of {} across parameter selections'.format(subset.region_labels[0]), fontsize=30, fontweight='bold', pad=16)
    ax0 = sns.heatmap(node1, cmap='bone_r', cbar=False)
    ax0.set_xlabel('Noise', fontsize=22, fontweight='bold', labelpad=10)
    ax0.set_ylabel('Global Coupling', fontsize=22, fontweight='bold', labelpad=10)
    ax0.invert_yaxis()

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_title('Weight of {} across parameter selections'.format(subset.region_labels[1]), fontsize=30, fontweight='bold', pad=16)
    ax1 = sns.heatmap(node2, cmap='bone_r', cbar=False)
    ax1.set_xlabel('Noise', fontsize=22, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Global Coupling', fontsize=22, fontweight='bold', labelpad=10)
    ax1.invert_yaxis()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Weight of {} across parameter selections'.format(subset.region_labels[2]), fontsize=30, fontweight='bold', pad=16)
    ax2 = sns.heatmap(node3, cmap='bone_r', cbar=False)
    ax2.set_xlabel('Noise', fontsize=22, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Global Coupling', fontsize=22, fontweight='bold', labelpad=10)
    ax2.invert_yaxis()

    # ax3 = fig.add_subplot(gs[1, 1])
    # ax3.set_title('Weight of {} across parameter selections'.format(subset.region_labels[3]), fontsize=30, fontweight='bold', pad=16)
    # ax3 = sns.heatmap(node4, cmap='bone_r', cbar=False)
    # ax3.set_xlabel('Noise', fontsize=22, fontweight='bold', labelpad=10)
    # ax3.set_ylabel('Global Coupling', fontsize=22, fontweight='bold', labelpad=10)
    # ax3.invert_yaxis()
    #
    # ax4 = fig.add_subplot(gs[0, 2])
    # ax4.set_title('Weight of {} across parameter selections'.format(subset.region_labels[0]), fontsize=30, fontweight='bold', pad=16)
    # ax4 = sns.heatmap(node5, cmap='bone_r', cbar=False)
    # ax4.set_xlabel('Noise', fontsize=22, fontweight='bold', labelpad=10)
    # ax4.set_ylabel('Global Coupling', fontsize=22, fontweight='bold', labelpad=10)
    # ax4.invert_yaxis()
    fig.savefig(os.path.join(ssdiFigures, "SJ3D_withlink_3node_2MACRO_ps_gc-noise_parameterSweep_and_macroWeightingforALLNODES.svg"))

    plt.show()










