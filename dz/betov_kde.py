"""
Implement 1D KDE from Betov et al., 2010, after R and Matlab code from 
https://web.maths.unsw.edu.au/~zdravkobotev/. Also uses code elements from
IsoplotR (Vermeesch,2018).
"""

import numpy as np
from scipy import optimize,stats

def kde(data,kde_min=None,kde_max=None):
    
    #fsolve will fail if using another n.
    n=512
    
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
    
    a = discrete_cosine_transform(counts_percent)
    
    # Compute bandwidth
    
    I = np.arange(1,n)**2
    a2 = (a[1:n+1]/2)**2
    
    bnds = (0,0.1)
    
    # This struggles and may need revision.
    t_star = optimize.fsolve(fixed_point,0,args=(unique_datapoints,I,a2))
    
    print(t_star)
    
    # Smoothing of DCT with t_star
    n_iterations = np.arange(0,n)
    
    a_t = a*np.exp(-(n_iterations)**2 * np.pi**2 * t_star/2)
    
    # Do inverse DCT
    density = inverse_dct(a_t)/kde_range
    
    # This is related to rescaling of data
    bandwidth = np.sqrt(t_star)*kde_range
    
    return(xgrid[0:-1],density,bandwidth[0])
    
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

def inverse_dct(data):
    data_length = len(data)
    entries = np.arange(0,data_length)
    
    weights = data_length * np.exp(1j*entries*np.pi/(2*data_length))
    
    # Note, in R script, this has to be divided by data_length, which gives
    # same output as below.
    data_transformed = np.real(np.fft.ifft(weights*data))
    
    # Reorder the data
    output = np.zeros(data_length)
    
    output[0:data_length:2] = data_transformed[0:int(data_length/2)]
    output[1:data_length:2] = data_transformed[data_length-1:int(data_length/2-1):-1]
    
    return(output)
    

def fixed_point(t,N,I,a2):
    """
    From Betov code:
    this implements the function t-zeta*gamma^[l](t)

    No absolute value in this version (after Betov matlab and Vermeesch,
    but Betov R version has it)    
    """
    l = 7
    f = 2 * (np.pi**(2*l)) * np.sum((I**l)*a2*np.exp(-I*np.pi**2*t))
    
    for s in np.arange(l-1,1,-1):
        K0 = np.prod(np.arange(1,2*s,2))/np.sqrt(2*np.pi)
        const = (1+(1/2)**(s+1/2))/3
        time = (2*const*K0/N/f)**(2/(3+2*s))
        f = 2*np.pi**(2*s)*np.sum(I**s*a2*np.exp(-I*np.pi**2*time))
    
    out = t - (2*N*np.sqrt(np.pi)*f)**(-2/5)
    return(out)


def pilot_density(data,bandwidth):
    data_length = len(data)
    
    density = np.zeros(data_length)
    for k,x in enumerate(data):
        density = density + stats.norm.pdf(data,loc=x,scale=bandwidth)
    
    return(density)

def adaptive_kde(data,bandwidth):
    
    density = pilot_density(data,bandwidth)
    density_mean = stats.gmean(density)
    
    
    factor = np.sqrt(density_mean/density)
    
    return(factor)
        
    