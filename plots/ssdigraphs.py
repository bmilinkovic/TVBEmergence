import os
import time
import datetime

import numpy as np
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import networkx as nx
import pandas as pd
from scipy import stats


def plot_macro_gc(edge_weights, node_weights, trials):

    for i in range(trials):            # the range is over the amount of simulations performed.
        subset = pd.DataFrame(edge_weights[:, edge_weights[i]:edge_weights[i+1]])
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
        ax1.set_title("GC-graph of an coupled {0}-node SJ3D model".format(int(len(subset))), fontsize=30, fontweight='bold', pad=16)
        pos = nx.spring_layout(G, seed=7)
        nx.draw_networkx_nodes(G, pos, node_size=1600, node_color='lightgray', linewidths=1.0, edgecolors='black')
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights,node_size=1600, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r)
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica")
        edge_labels = dict([((u, v,), f"{d['weight']:.2f}") for u, v, d in G.edges(data=True)])


        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_title("2-Macro on GC-graph of coupled {0}-node SJ3D model".format(int(len(subset))), fontsize=30, fontweight='bold', pad=16)
        pos = nx.spring_layout(G, seed=7)
        nx.draw_networkx_nodes(G, pos, node_size=1600, node_color=node_weights[:, i], cmap=plt.cm.Blues, linewidths=1.0, edgecolors='black')
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights, node_size=1600, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r)
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica")
        edge_labels = dict([((u, v,), f"{d['weight']:.1f}") for u, v, d in G.edges(data=True)])

        return fig



def plot_dd(dd_values, macrosize):
    """
    Plots the macro DD values in parameter space.

    Args:
        dd_values (numpy.ndarray): A 3D array of shape (n_coupling, n_noise, n_macro) containing the macro DD values.
        macrosize (list): A list of integers indicating the number of nodes in each macro.

    Returns:
        matplotlib.figure.Figure: The resulting figure object.
    """

    coupling = [str(round(float(x), 2)) for x in 10**np.r_[-2:-0.7:20j]]
    noise = [str(round(float(x), 3)) for x in 10**np.r_[-6:-1:20j]]

    rows = 1
    cols = dd_values.shape[2]

    gs = GridSpec(rows, cols, width_ratios=[1, 1, 1, 1.2])  # <-- Add spacing between subplots and set equal width ratios

    cmap = sns.color_palette("YlOrBr", as_cmap=True)

    fig = plt.figure(figsize=(cols*6.5, rows*8.0))  # <-- Adjust the size here
    fig.suptitle("Macro DD values in parameter space", fontsize=16, fontweight="bold", y=0.90)

    # Set a global x-axis label
    fig.text(0.5, 0.1, 'Noise', ha='center', fontsize=14, fontweight='bold')

    # Set a global y-axis label
    fig.text(0.07, 0.5, 'Global Coupling', va='center', rotation='vertical', fontsize=14, fontweight='bold')

    vmin = np.min(dd_values)
    vmax = np.max(dd_values)

    cbar_ax = None  # Initialize colorbar axis

    for i in range(dd_values.shape[2]):

        node = pd.DataFrame(dd_values[:,:,i])
        node.columns = [noise]
        node.index = [coupling]
        ax = fig.add_subplot(gs[i])
        ax.set_title(f'{macrosize[i]}-macro', fontsize=14, fontweight='bold', pad=8)
        ax = sns.heatmap(node, cmap=cmap, vmin=vmin, vmax=vmax, cbar=i==cols-1, linecolor='black', linewidths=.6, square=True, cbar_kws={'shrink': 0.5, 'label': 'DD Value'})
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks(np.arange(0, len(noise)+1, 2))
        ax.set_yticks(np.arange(0, len(coupling)+1, 2))
        ax.set_xticklabels(noise[::2] + [noise[-1]], fontsize=14, rotation=45)
        ax.set_yticklabels(coupling[::2] + [coupling[-1]], fontsize=14, rotation=45)
        ax.invert_yaxis()

    cbar = ax.collections[0].colorbar  # Get the colorbar
    cbar.ax.tick_params(labelsize=14)  # Set the tick label size
    cbar.set_label('DD Values', fontsize=16)

    # Save the figure as PNG and EPS
    fig.savefig('dd_values_{}macro.png'.format(macrosize[i]), dpi=300, bbox_inches='tight')
    fig.savefig('dd_values_{}macro.eps'.format(macrosize[i]), format='eps', dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_nodeweights_all(node_weights, macrosize, regions = None):


    region_labels = [x for x in range(node_weights.shape[0])]
    # initialise values:
    region_labels = regions.region_labels
    n_nodes = np.size(region_labels)

    fig = plt.figure(figsize=(7, 7))
    gs = GridSpec(nrows=1, ncols=1)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title('Node-weight by contribution to {}-MACRO for each Parameter Regime'.format(macrosize), fontsize=12, fontweight='bold', pad=16)
    ax0 = sns.heatmap(node_weights, yticklabels=np.flip(region_labels), cmap='bone_r')
    ax0.set_xlabel('Parameter Sweep Run', fontsize=12, fontweight='bold', labelpad=10)
    ax0.set_ylabel('Brain Region (Node)', fontsize=12, fontweight='bold', labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, fontweight='bold', rotation=90)

    return fig




def plot_nodeweights_individual(node_weights, macrosize, regions = None):

    coupling = [str(round(float(x), 2)) for x in 10**np.r_[-2:-0.7:20j]]
    noise = [str(round(float(x), 3)) for x in 10**np.r_[-6:-1:20j]]

    region_labels = [x for x in range(node_weights.shape[0])]

    nodes = node_weights.shape[0]
    cols = 2
    rows = (nodes + 1) // cols
    gs = GridSpec(rows, cols)  # <-- Add spacing between subplots

    fig = plt.figure(figsize=(11, 14))  # <-- Adjust the size here
    fig.suptitle("Region distance from {}-macro in parameter space".format(macrosize), fontsize=16, fontweight="bold")
    
    # Set a global x-axis label
    fig.text(0.5, 0.09, 'Noise', ha='center', fontsize=14, fontweight='bold')

    # Set a global y-axis label
    fig.text(0.09, 0.5, 'Global Coupling', va='center', rotation='vertical', fontsize=14, fontweight='bold')

    for i, nodes in enumerate(node_weights):

        node = pd.DataFrame(nodes.reshape(20, 20))
        node.columns = [noise]
        node.index = [coupling]
        ax = fig.add_subplot(gs[i])
        ax.set_title('Node {}'.format(str(region_labels[i]+1)), fontsize=14, fontweight='bold', pad=8)
        ax = sns.heatmap(node, cmap='bone_r', cbar=False, linecolor='black', linewidths=.6, square=True)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels(noise, fontsize=8, rotation=45)
        ax.set_yticklabels(coupling, fontsize=8, rotation=45)
        ax.invert_yaxis()
    
    cbar_ax = fig.add_axes([0.62, 0.24, 0.2, 0.02])  # Adjust the position and size of the colorbar
    cbar = fig.colorbar(ax.collections[0], cax=cbar_ax, orientation='horizontal')  # Use any of the heatmaps for the colorbar
    cbar.ax.tick_params(labelsize=12)  # Set the tick label size
    cbar.set_label('Subspace distance to {}-macro'.format(macrosize), fontsize=16)  # Set the label for the colorbar
    # Set the tick labels for 0 and 1
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(['0', '0.25', '0.5', '0.75', '1'])
    plt.show()
    return fig



def plot_gc(edge_weights):
    # the range is over the amount of simulations performed.
    subset = pd.DataFrame(edge_weights)
    # subset.columns = ['0', '1', '2']
    # subset.index = ['0','1','2']
    G = nx.from_pandas_adjacency(subset, create_using=nx.MultiDiGraph)
    G.remove_edges_from(list(nx.selfloop_edges(G)))                    # remove self edges
    node_labels = {i: i+1 for i in range(len(subset))}
    nx.relabel_nodes(G, node_labels, copy=False)
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items()) # Extracts the edges and their corresponding weights into different tuples
    
    n_cols = 2
    n_rows = 1
    # Calculate the width and height ratios
    width_ratios = [1.2] * n_cols
    height_ratios = [1] * n_rows
    # PLOTTING THE PWCGC MATRIX and GRAPH.
    fig = plt.figure(figsize=(13, 6)) # Change the figsize to adjust the size of the graph
    gs = GridSpec(nrows=n_rows, ncols=n_cols, width_ratios=width_ratios, height_ratios=height_ratios)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title("{0}-variable Granger-causal Matrix".format(int(len(subset))), fontsize=20, fontweight='bold', pad=26)
    mask = subset == 0
    sns.heatmap(subset, cmap=mpl.cm.bone_r, center=0.5, linewidths=.6, linecolor='black',annot=True, cbar_kws={'label': 'G-causal estimate values', 'shrink': 0.5, 'orientation': 'vertical'}, ax=ax0, mask=mask)
    cbar = ax0.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('G-causal estimate values', fontsize=14, fontweight='bold', labelpad=10)
    # Modify the x and y tick labels to start from 1 instead of 0
    ax0.set_xticklabels([str(int(x)+1) for x in ax0.get_xticks()])
    ax0.set_yticklabels([str(int(y)+1) for y in ax0.get_yticks()])
    ax0.invert_yaxis()
    ax0.set_aspect('equal')
    ax0.tick_params(labelsize=16)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title("{0}-variable Granger-causal graph".format(int(len(subset))), fontsize=20, fontweight='bold', pad=16)
    pos = nx.spring_layout(G, seed=7)
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightgray', linewidths=1.0, edgecolors='black', ax=ax1) # Change the node_size to adjust the size of the nodes
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights,node_size=800, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r, ax=ax1) # Change the node_size to adjust the size of the nodes
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica", ax=ax1)
    edge_labels = dict([((u, v,), f"{d['weight']:.2f}") for u, v, d in G.edges(data=True)])
    ax1.set_aspect('equal')
    ax1.tick_params(labelsize=16)

    ax1.axis('off')

    plt.tight_layout()
    # Save the figure as an eps and a png file with the name corresponding to the size of the granger causal matrix
    filename = f"{int(len(subset))}-variable"
    fig.savefig(os.path.join(os.getcwd(), f"{filename}.eps"), format='eps')
    fig.savefig(os.path.join(os.getcwd(), f"{filename}.png"), format='png')

    return fig

def plot_nweights(eweights, nweights, macrosize, opt_number):
        
        subset = pd.DataFrame(eweights)
        # subset.columns = ['0', '1', '2']
        # subset.index = ['0','1','2']
        G = nx.from_pandas_adjacency(subset, create_using=nx.MultiDiGraph)
        G.remove_edges_from(list(nx.selfloop_edges(G)))                    # remove self edges
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items()) # Extracts the edges and their corresponding weights into different tuples

        # PLOTTING THE PWCGC MATRIX, GRAPH AND MACRO PROJECTION ON GRAPH.
        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(nrows=1, ncols=1)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_title("{0}-Macro on GC-graph of coupled {1}-node model".format(int(macrosize), int(len(subset))), fontsize=18, fontweight='bold', pad=16)
        pos = nx.spring_layout(G, seed=7)
        nx.draw_networkx_nodes(G, pos, node_size=1600, node_color=nweights[:,opt_number], cmap=plt.cm.Blues, linewidths=1.0, edgecolors='black') # nweights[:,0] will plot the optimal projection of the first macro variable on the graph.
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights, node_size=1600, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r)
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica")
        edge_labels = dict([((u, v,), f"{d['weight']:.1f}") for u, v, d in G.edges(data=True)])

        return fig


def plot_optp(preopthist, preoptdist):

    n_cols = 2
    n_rows = 1
    # Calculate the width and height ratios
    width_ratios = [1.2] * n_cols
    height_ratios = [1] * n_rows


    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(nrows=n_rows, ncols=n_cols, width_ratios=
                  width_ratios, height_ratios=height_ratios)
    # Set the width and height ratios for the GridSpec
    #gs.update(width_ratios=width_ratios, height_ratios=height_ratios)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Pre-optimisation History', fontweight='bold', fontsize=18)
    ax1.set_ylabel('Dynamical Dependence', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Iterations', fontweight='bold', fontsize=14)
    for i in range(len(preopthist)):
        ax1 = sns.lineplot(data=preopthist[i][0][:, 0], legend=False, dashes=False, palette='bone_r', linewidth=0.6)

    cmap = sns.color_palette("bone_r", as_cmap=True)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Local-Optima Distances', fontweight='bold', fontsize=14)
    ax2 = sns.heatmap(preoptdist['goptp'], cmap=cmap, center=np.max(preoptdist['goptp'])/2, cbar_kws={'label': 'Orthogonality of subspaces'})
    ax2.set_xlabel('Optimisation run', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Optimisation run', fontweight='bold', fontsize=14)
    ax2.set_xticklabels(ax2.get_xmajorticklabels(), fontsize = 8)
    ax2.set_yticklabels(ax2.get_ymajorticklabels(), fontsize = 8)
    ax2.invert_yaxis()

    return fig


def plot_opto(opthist, optdist):

    fig = plt.figure(figsize=(10,10))
    gs = GridSpec(nrows=1, ncols=2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Optimisation History: ', fontweight='bold', fontsize=18)
    ax1.set_ylabel('Dynamical Dependence', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Iterations', fontweight='bold', fontsize=14)
    ax1.set_xscale('log')
    for i in range(len(opthist)):
        ax1 = sns.lineplot(data=opthist[i][0][:, 0], legend=False, dashes=False, palette='bone_r')

    cmap = sns.color_palette("bone_r", as_cmap=True)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Local-Optima Distances: ', fontweight='bold', fontsize=18)
    ax2 = sns.heatmap(optdist['gopto'], cmap=cmap, center=np.max(optdist['gopto'])/2, cbar_kws={'label': 'Othogonality of subspaces'})
    ax2.set_xlabel('Optimisation run', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Optimisation run', fontweight='bold', fontsize=14)
    ax2.set_xticklabels(ax2.get_xmajorticklabels(), fontsize = 8)
    ax2.set_yticklabels(ax2.get_ymajorticklabels(), fontsize = 8)
    ax2.invert_yaxis()

    return fig


def plot_connectivity(subset, show_figure=True, annot=False):
    """Plots structural connectivity and tracts matrix of 'network'.

    Args:
        subset (object): A `Subset` object containing the structural connectivity and tract lengths data.
        show_figure (bool): A flag to indicate whether to display the figure or not. Default is True.
        annot (bool): A flag to indicate whether to annotate the heatmap or not. Default is False.

    Returns:
        None.

    This function generates a figure with two subplots, one for the structural connectivity matrix and one for the tracts matrix.
    The structural connectivity matrix shows the coupling strength between each pair of regions of interest (ROIs) in the network,
    while the tracts matrix shows the distance (in millimeters) between each pair of ROIs. The ROIs are labeled on both axes of
    each matrix. The matrices are generated using the `sns.heatmap` function from the Seaborn library, with a reversed "bone"
    color palette. The matrices are also annotated with their respective values if `annot` is True. The generated figure is saved in the
    "../results/connectivity/" directory as both a PNG and an EPS file.

    """

    weights = subset.weights
    tracts = subset.tract_lengths
    regionLabels = subset.region_labels

    f = plt.figure(figsize=(14, 6))
    gs = f.add_gridspec(1, 2, wspace=0.35) # add wspace parameter to adjust horizontal space between the two axes

    ax1 = f.add_subplot(gs[0, 0])
    ax1.set_title('Weights Matrix', fontsize=22, fontweight='bold', pad=18)
    cmap = mpl.cm.bone_r  # setting colour pallette to "bone" but reversed
    weightsConn = sns.heatmap(weights.T, cmap=cmap,
                          center=1.5, cbar_kws={'shrink': 0.6}, linewidths=.6, xticklabels=np.flip(regionLabels),
                          yticklabels=regionLabels, annot=annot, square=True, cbar=True)
    weightsConn.set_xlabel('To (ROIs)', fontsize=18, labelpad=10, fontweight='bold')
    weightsConn.set_ylabel('From (ROIs)', fontsize=18, labelpad=8, fontweight='bold')
    weightsConn.tick_params(axis='both', which='major', labelsize=12) # added parameter for font size of xtick labels and ytick labels
    cbar = weightsConn.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14) # set fontsize of colorbar ticks
    cbar.set_ticks([0, 1, 2, 3]) # set colorbar ticks
    cbar.set_label('Coupling Strength', fontsize=16) # set colorbar label
    

    ax2 = f.add_subplot(gs[0, 1])
    ax2.set_title('Tracts Matrix', fontsize=22, fontweight='bold', pad=18)
    cmap = mpl.cm.bone_r  # setting colour pallette to "bone" but reversed
    tractsConn = sns.heatmap(tracts.T, cmap=cmap,
                         center=np.max(tracts)/2, cbar_kws={'shrink': 0.6}, linewidths=.6, xticklabels=np.flip(regionLabels),
                         yticklabels=regionLabels, annot=annot, square=True)
    tractsConn.set_xlabel('To (ROIs)', fontsize=18, labelpad=10, fontweight='bold')
    tractsConn.set_ylabel('From (ROIs)', fontsize=18, labelpad=8, fontweight='bold')
    tractsConn.tick_params(axis='both', which='major', labelsize=12) # added parameter for font size of xtick labels and ytick labels
    cbar = tractsConn.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14) # set fontsize of colorbar ticks
    cbar.set_label('Distance (mm)', fontsize=16) # set colorbar label
    
    if show_figure:
        plt.show()

def plot_wholebrain_connectivity(subset, show_figure=True, annot=False):
    """Plots structural connectivity and tracts matrix of 'network'.

    Args:
        subset (object): A `Subset` object containing the structural connectivity and tract lengths data.
        show_figure (bool): A flag to indicate whether to display the figure or not. Default is True.
        annot (bool): A flag to indicate whether to annotate the heatmap or not. Default is False.

    Returns:
        None.

    This function generates a figure with two subplots, one for the structural connectivity matrix and one for the tracts matrix.
    The structural connectivity matrix shows the coupling strength between each pair of regions of interest (ROIs) in the network,
    while the tracts matrix shows the distance (in millimeters) between each pair of ROIs. The ROIs are labeled on both axes of
    each matrix. The matrices are generated using the `sns.heatmap` function from the Seaborn library, with a reversed "bone"
    color palette. The matrices are also annotated with their respective values if `annot` is True. The generated figure is saved in the
    "../results/connectivity/" directory as both a PNG and an EPS file.

    """

    weights = subset.weights
    tracts = subset.tract_lengths
    regionLabels = subset.region_labels

    f = plt.figure(figsize=(39.2, 16.8)) # increase width and height by 40%
    gs = f.add_gridspec(1, 2, wspace=0.35) # add wspace parameter to adjust horizontal space between the two axes

    ax1 = f.add_subplot(gs[0, 0])
    ax1.set_title('Weights Matrix', fontsize=22, fontweight='bold', pad=18)
    cmap = mpl.cm.bone_r  # setting colour pallette to "bone" but reversed
    weightsConn = sns.heatmap(weights.T, cmap=cmap,
                          center=1.5, cbar_kws={'shrink': 0.6}, linewidths=.6, xticklabels=np.flip(regionLabels),
                          yticklabels=regionLabels, annot=annot, square=True, cbar=True)
    weightsConn.set_xlabel('To (ROIs)', fontsize=18, labelpad=10, fontweight='bold')
    weightsConn.set_ylabel('From (ROIs)', fontsize=18, labelpad=8, fontweight='bold')
    weightsConn.tick_params(axis='both', which='major', labelsize=12) # added parameter for font size of xtick labels and ytick labels
    cbar = weightsConn.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14) # set fontsize of colorbar ticks
    cbar.set_ticks([0, 1, 2, 3]) # set colorbar ticks
    cbar.set_label('Coupling Strength', fontsize=16) # set colorbar label
    

    ax2 = f.add_subplot(gs[0, 1])
    ax2.set_title('Tracts Matrix', fontsize=22, fontweight='bold', pad=18)
    cmap = mpl.cm.bone_r  # setting colour pallette to "bone" but reversed
    tractsConn = sns.heatmap(tracts.T, cmap=cmap,
                         center=np.max(tracts)/2, cbar_kws={'shrink': 0.6}, linewidths=.6, xticklabels=np.flip(regionLabels),
                         yticklabels=regionLabels, annot=annot, square=True)
    tractsConn.set_xlabel('To (ROIs)', fontsize=18, labelpad=10, fontweight='bold')
    tractsConn.set_ylabel('From (ROIs)', fontsize=18, labelpad=8, fontweight='bold')
    tractsConn.tick_params(axis='both', which='major', labelsize=12) # added parameter for font size of xtick labels and ytick labels
    cbar = tractsConn.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14) # set fontsize of colorbar ticks
    cbar.set_label('Distance (mm)', fontsize=16) # set colorbar label
    
    if show_figure:
        plt.show()


