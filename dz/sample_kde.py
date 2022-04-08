"""
Test for sample dataset from IsoplotR on botev.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import botev

#%% Test Botev Algorithim

data = np.genfromtxt('sample_data.csv',delimiter=',')

# Get each bandwidth
grid,density,bandwidth = botev.py_kde(data)
bandwidth_botev_r = botev.botev_r(data)
bandwidth_vermeesch = botev.vermeesch_r(data)

print(bandwidth,bandwidth_botev_r,bandwidth_vermeesch)

fig,axs = plt.subplots(5,sharex=True,sharey=True,dpi=300,
                       figsize=(8.5,11))

std = data.std()
sns.kdeplot(data,ax=axs[0],bw_method=bandwidth/std)
axs[0].set_title('PyBotev - Seaborn')

axs[1].plot(grid,density)
axs[1].set_title('PyBotev - Direct')

sns.kdeplot(data,ax=axs[2],bw_method=bandwidth_botev_r/std)
axs[2].set_title('Botev R - Seaborn')

sns.kdeplot(data,ax=axs[3],bw_method=bandwidth_vermeesch/std)
axs[3].set_title('Vermeesch R - Seaborn')

#%% Test Adaptive Bandwidth

# Currently using the Seaborn weights feature, don't think this is correct though

bw_weights = botev.adaptive_kde(data,bandwidth)

sns.kdeplot(data,ax=axs[4],bw_method=bandwidth/std,weights=bw_weights/std)
axs[4].set_title('Seaborn Adaptive - PyBotev')

plt.tight_layout()