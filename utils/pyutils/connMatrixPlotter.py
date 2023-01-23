import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np


def connMatrixPlotter(network):

    # plots structural connectivity and tracts matrix of 'network'

    weights = network.weights
    tracts = network.tract_lengths
    regionLabels = network.region_labels

    f = plt.figure(figsize=(21, 8))
    gs = f.add_gridspec(1, 2)

    ax1 = f.add_subplot(gs[0, 0])
    ax1.set_title('Weights Matrix', fontsize=30, fontweight='bold', pad=15)
    cmap = mpl.cm.bone_r  # setting colour pallette to "bone" but reversed
    weightsConn = sns.heatmap(np.flip(weights, axis=0), cmap=cmap,
                          cbar_kws={'label': 'Coupling Strength between Brain Regions'},
                          center=np.max(weights)/2, linewidths=.6, xticklabels=regionLabels,
                          yticklabels=np.flip(regionLabels), annot=True)
    weightsConn.set_xlabel('From (brain regions)', fontsize=20, labelpad=10)
    weightsConn.set_ylabel('To (brain regions)', fontsize=20, labelpad=8)

    ax2 = f.add_subplot(gs[0, 1])
    ax2.set_title('Tracts Matrix', fontsize=30, fontweight='bold', pad=15)
    cmap = mpl.cm.bone_r  # setting colour pallette to "bone" but reversed
    tractsConn = sns.heatmap(np.flip(tracts, axis=0), cmap=cmap,
                         cbar_kws={'label': 'Length (mm)'},
                         center=np.max(tracts)/2, linewidths=.6, xticklabels=regionLabels,
                         yticklabels=np.flip(regionLabels), annot=True)
    tractsConn.set_xlabel(' From (brain regions)', fontsize=20, labelpad=10)
    tractsConn.set_ylabel('To (brain regions)', fontsize=20, labelpad=8)
