#%%
import numpy as np
import scipy.io as sio
import os

from plots.ssdigraphs import plot_phiid_matrix
import matplotlib.pyplot as plt

#%%
# Load data

data = 'results/phiid/MMI/'
figure_results = 'results/phiid/MMI/figures/'

if not os.path.exists(figure_results):
    os.makedirs(figure_results)


#%% load datafiles from data

# Get a list of all files in the data directory
files = os.listdir(data)

# Filter the list to only include files with 'sts' in the filename
filtered_files = [f for f in files if 'rtr' in f]

#%% Load the datafiles
for f in filtered_files:
    data_path = os.path.join(data, f)

    # Load the MATLAB data file
    mat_data = sio.loadmat(data_path)

    # Access the data using the variable names in the MATLAB file
    data_matrix = mat_data['rtr_mat']
    data_matrix = data_matrix / np.max(np.abs(data_matrix))
    
    # Extract the filename from the filtered_files list
    #filename = f.split('.')[0]

    # Plot the sts matrix with the filename as the title
    plot_phiid_matrix(data_matrix, f, show_figure=False)
    plt.savefig(os.path.join(figure_results, f[:-4] + '.png'))
    
# %% plot group averaged connectivity matrices
all_conn = []
for file in files:
    if 'W' in file and 'rtr' in file:
        mat_data = sio.loadmat(os.path.join(data, file))
        data_matrix = mat_data['rtr_mat']
        data_matrix = data_matrix / np.max(np.abs(data_matrix))
        all_conn.append(data_matrix)

avg_matrix = np.average(all_conn, axis=0)
np.save('wake_rtr_avg_matrix.npy', avg_matrix)

fig = plot_phiid_matrix(avg_matrix, 'Wake Averaged Red-to-Red matrix')
plt.savefig(os.path.join(figure_results, 'Wake_averaged_rtr' + '.png'))
# %%
