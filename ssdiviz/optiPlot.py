import os
import time

import numpy as np
import scipy.io as sio

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import pandas as pd




#%%

ssdiPreopt = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/ssdiDataMATLAB/preoptData'
ssdiOpt = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/ssdiDataMATLAB/optData'

ssdiOptfile = 'optDD_mdim3_plotting data.mat'
ssdiPreoptfile = 'preoptDD_mdim3_plotting data.mat'

preopt = sio.loadmat(os.path.join(ssdiPreopt, ssdiPreoptfile))
preopthist = preopt['preoptimisation_history']
opt = sio.loadmat(os.path.join(ssdiOpt, ssdiOptfile))
opthist = opt['optimisation_history']




# GRAPHING OPTMISATION AND PREOPTIMISATION HISTORIES WITH CORRESPONDING LOCAL-OPTIMA DISTANCES

fig = plt.figure(figsize=(20,20))
gs = GridSpec(nrows=2, ncols=2)

ax0 = fig.add_subplot(gs[0, 0])
ax0.set_title('Pre-optimisation History: ', fontweight='bold', fontsize=11)
ax0.set_ylabel('Dynamical Dependence', fontsize=8)
ax0.set_xlabel('Iterations', fontsize=8)
ax0.set_xscale('log')
for i in range(len(preopthist[0])):
    ax0 = sns.lineplot(data=preopthist[0, i], legend=False, dashes=False, palette='bone_r')


ax1 = fig.add_subplot(gs[0, 1])
ax1.set_title('Optimisation History: ', fontweight='bold', fontsize=11)
ax1.set_ylabel('Dynamical Dependence', fontsize=8)
ax1.set_xlabel('Iterations', fontsize=8)
ax1.set_xscale('log')
for i in range(len(opthist[0])):
    ax0 = sns.lineplot(data=opthist[0, i], legend=False, dashes=False, palette='bone_r')



cmap = sns.color_palette("bone_r", as_cmap=True)
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('Local-Optima Distances: ', fontweight='bold', fontsize=11)
ax2 = sns.heatmap(preopt['goptp'], cmap=cmap, center=np.max(preopt['goptp'])/2)
ax2.set_xlabel('Optimisation runs', fontsize=8)
ax2.set_xticklabels(ax2.get_xmajorticklabels(), fontsize = 8)
ax2.set_yticklabels(ax2.get_ymajorticklabels(), fontsize = 8)
ax2.invert_yaxis()


ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title('Local-Optima Distances: ', fontweight='bold', fontsize=11)
ax3 = sns.heatmap(opt['gopto'], cmap=cmap, center=np.max(opt['gopto'])/2)
ax3.set_xlabel('Optimisation runs', fontsize=8)
ax3.set_xticklabels(ax3.get_xmajorticklabels(), fontsize = 8)
ax3.set_yticklabels(ax3.get_ymajorticklabels(), fontsize = 8)
ax3.invert_yaxis()

fig.show()