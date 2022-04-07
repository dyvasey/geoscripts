"""
Implement 1D KDE from Betov et al., 2010, after R code from 
https://web.maths.unsw.edu.au/~zdravkobotev/
"""

import numpy as np

def kde(data,n=2**14,kde_min=None,kde_max=None):
    
    # Define data min/max and range
    data_min = np.min(data)
    data_max = np.max(data)
    data_range = data_max - data_min
    
    # Set default min/max if not specified
    if kde_min is None:
        # NOTE, in current function on Betov website, it's range/2,
        # but all documentation and other implementations say range/10.
        kde_min = data_min - data_range/10
    
    if kde_max is None:
        kde_max = data_max + data_range/10
    
    # Make n a power of two if needed
    if not np.log2(n).is_integer():
        print(n,' is not a power of 2')
        n = 2**np.ceil(np.log2(n))
        print('New n: ',n)
    
    # Set up grid
    kde_range = kde_max - kde_min
    
    # NOTE: dx following Botev's R script and Vermeesch's R script
    # but Botev matlab script uses n-1
    dx = kde_range/n
    xgrid = np.arange(kde_min,kde_max+dx,dx)
    
    # Get number of unique datapoints
    unique_datapoints = len(np.unique(data))
    
    # Bin data uniformly using grid
    counts,bins = np.histogram(data,bins=xgrid)
    
    # Get fraction of each bin relative to unique datapoints
    # Note: not sure why/if this is necessary
    counts_fraction = counts/unique_datapoints
    
    # Normalize to a percentage
    counts_percent = counts_fraction/np.sum(counts_fraction)
    
    dct = discrete_cosine_transform(counts_percent)
    
def discrete_cosine_transform(data):
    data_length = len(data)
    weight_initial = np.array([1])
    
    entries = np.arange(1,data_length)
    remaining_weights = 2*np.exp(-1j*entries*np.pi/(2*data_length))
    
    weights = np.concatenate([weight_initial,remaining_weights])
    
    # Reorder the data
    data1 = data[0:data_length:2]
    data2 = data[data_length:0:-2]
    data_reorder = np.concatenate([data1,data2])
    
    # Multiply FFT by weights
    data_transformed = np.real(weights*np.fft.fft(data_reorder))

    return(data_transformed)