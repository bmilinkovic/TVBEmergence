import matplotlib.pyplot as plt
from scipy.stats import ranksums
import numpy as np
import scipy.io as sio
import os
import seaborn as sns



def ddstat(x, y):
    """ Calculates the Wilcoxon Rank-Sum Test between vector valued variable x
    and vector valued variable y

    _____________________
    Output:
    stat: the statistic value of the test
    pval: the p-value
    """
    stat, pval = ranksums(x, y, alternative='greater')
    return stat, pval


def ddstatMultivariate(x, y):
    stat, pval = [], []
    for i in range(np.size(x, axis=0)):
        st, p = ddstat(x[i, :], y[i, :])
        stat.append(st)
        pval.append(p)
    return [stat, pval]


# Unit test
# Load in data.
nw_coupled_dir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/ssdiDataMATLAB/SJ3D_5node_nodelay_ps_gc-noise/ssdiData/'
nw_uncoupled_dir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/ssdiDataMATLAB/SJ3D_NOCONN_5node_nodelay_ps_gc-noise/ssdiData/'

nw_coupled_file = 'SJ3D_AIC_3MACRO_nodeWeights_parametersweep_noise_gc.mat'
nw_uncoupled_file = 'SJ3D_NOCONN_AIC_3MACRO_nodeWeights_parametersweep_noise_gc.mat'

nw_coupled = os.path.join(nw_coupled_dir, nw_coupled_file)
nw_uncoupled = os.path.join(nw_uncoupled_dir, nw_uncoupled_file)

coupled = sio.loadmat(nw_coupled)
uncoupled = sio.loadmat(nw_uncoupled)
coupled = coupled['maximalNodeWeights']
uncoupled = uncoupled['maximalNodeWeights']

stat, pval = ddstatMultivariate(coupled, uncoupled)


fig, ax = plt.subplots(figsize=(20,10))
plot = ax.bar(['rTCI', 'rA2', 'lM1', 'lTCC', 'rIP'], stat, color=['#008080'], edgecolor=['black'], alpha=0.7) # A bar chart
ax.set_title('Wilcoxon Rank-sum Test: H0: Node is not significantly implicated in the dynamics of a 3-Macro in the '
             'coupled regime in comparison to the uncoupled regime ', fontsize=14, fontweight='bold', pad=10)
ax.set_xlabel('Regions', fontsize=16, fontweight='bold', labelpad=10)
ax.set_xticklabels(labels=['rTCI', 'rA2', 'lM1', 'lTCC', 'rIP'], fontdict={'fontsize': 12, 'fontweight': 'bold'})
ax.set_ylabel('Wilcoxon Rank-Sum Z-score', fontsize=16, fontweight='bold')
ax.axhline(0, 0, color='black')

def autolabel(plot,  pval, xpos='center',):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.5, 'left': 0.7}  # x_txt = x + w*off

    for i, rect in enumerate(plot):
        height = rect.get_height()
        if pval[i] < 0.05:
            ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                'p = {0:.02g}*'.format(pval[i]), ha=ha[xpos], va='bottom')
        # else:
        #     ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
        #             'p = {0:.02g}'.format(pval[i]), ha=ha[xpos], va='bottom')

autolabel(plot, pval, "center")
plt.savefig('/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/results/ssdiFiguresPython/stats/SJ3D_3macro_ranksum.svg')
fig.show()

