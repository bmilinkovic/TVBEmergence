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
filtered_files = [f for f in files if 'sts' in f]

# Load the datafiles
for f in filtered_files:
    filepath = os.path.join(data, f)

    # Load the MATLAB data file
    mat_data = sio.loadmat(filepath)

    # Access the data using the variable names in the MATLAB file
    data = mat_data['sts_mat']
    
    # Extract the filename from the filtered_files list
    #filename = f.split('.')[0]

    # Plot the sts matrix with the filename as the title
    fig = plot_phiid_matrix(data, f)

    plt.savefig(os.path.join(figure_results, filename + '.png'))
    


# %%
