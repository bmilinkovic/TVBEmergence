#%%

from idtxl.bivariate_mi import BivariateMI
from idtxl.data import Data

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#%%

# Load example data

data = Data(np.random.rand(100, 2), dim_order='ps')
# %%

# Initialise analysis object and define settings
bmi = BivariateMI()
settings = {'cmi_estimator': 'JidtGaussianCMI',
            'max_lag_sources': 1,
            'min_lag_sources': 1}

# Run analysis

results = bmi.analyse_network(settings=settings, data=data)

# %%

# Print results

print('Bivariate MI: %.4f' % results['bivariate_mi'])
print('Sources: %s' % results['sources'])
print('Max. sources: %s' % results['max_lag_sources'])
print('Min. sources: %s' % results['min_lag_sources'])

# %%

# Plot results

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.heatmap(results['val_matrix'], cmap='viridis', ax=ax)
ax.set_title('Bivariate MI')
ax.set_xlabel('Target')
ax.set_ylabel('Source')
fig.tight_layout()
plt.show()
