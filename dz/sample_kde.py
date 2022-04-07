"""
Test for sample dataset from IsoplotR on betov_kde
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import betov_kde

#%% Test Betov Algorithim

data = np.genfromtxt('sample_data.csv',delimiter=',')

grid,density,bandwidth = betov_kde.kde(data)

fig,axs = plt.subplots(3,sharex=True,sharey=True)


std = data.std()
sns.kdeplot(data,ax=axs[0],bw_method=bandwidth/std)

axs[0].set_title('Seaborn')

axs[1].plot(grid,density)
axs[1].set_title('Betov Algorithim')

#%% Test Adaptive Bandwidth

# Currently using the weights feature, don't think this is correct though

bw_weights = betov_kde.adaptive_kde(data,bandwidth)

sns.kdeplot(data,ax=axs[2],bw_method=bandwidth/std,weights=bw_weights/std)
axs[2].set_title('Seaborn Adaptive')

plt.tight_layout()