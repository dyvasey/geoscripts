"""
Implement 1D KDE from Botev et al., 2010, after R and Matlab code from 
https://web.maths.unsw.edu.au/~zdravkobotev/ and R code from
IsoplotR (Vermeesch,2018).
"""
import os 

import numpy as np
import pandas as pd
from scipy import optimize,stats

import rpy2.robjects as ro

dz_dir = os.path.dirname(os.path.realpath(__file__))

def py_kde(data,kde_min=None,kde_max=None):
    """
    Pythonic implementation of Botev et al., 2010 algorithim. 
    
    Very hard to get to work with SciPy solvers, so currently deprecated 
    in preference of R-based solutions in other functions.

    Parameters:
        data: Array or series of ages
        kde_min: Minimum value for the KDE
        kde_max: Maximum value for the KDE
    
    Returns:
        xgrid: Grid for the KDE
        density: Densities at each point on the grid
        bandwidth: Bandwidth for each point on the grid
    """
    
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
    
    # This struggles and may need revision.
    t_star = optimize.minimize_scalar(fixed_point,bounds=(0,0.1),
                            args=(unique_datapoints,I,a2),tol=1e-22,
                            method='bounded').x
    
    print(t_star)
    
    # Smoothing of DCT with t_star
    n_iterations = np.arange(0,n)
    
    a_t = a*np.exp(-(n_iterations)**2 * np.pi**2 * t_star/2)
    
    # Do inverse DCT
    density = inverse_dct(a_t)/kde_range
    
    # This is related to rescaling of data
    bandwidth = np.sqrt(t_star)*kde_range
    
    return(xgrid[0:-1],density,bandwidth)
    
def discrete_cosine_transform(data):
    """
    Discrete cosine transform in support of py_kde function.

    Parameters:
        data: Data to be transformed
    
    Returns:
        data_transformed: Transformed data
    """
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
    """
    Inverse DCT in support of py_kde function

        Parameters:
        data: Data to be transformed
    
    Returns:
        data_transformed: Transformed data
    """
    data_length = len(data)
    entries = np.arange(0,data_length)
    
    weights = data_length * np.exp(1j*entries*np.pi/(2*data_length))
    
    # Note, norm='forward' needed to match output of R fft inverse
    data_transformed = np.real(np.fft.ifft(weights*data,norm='forward'))/data_length
    
    # Reorder the data
    output = np.zeros(data_length)
    
    output[0:data_length:2] = data_transformed[0:int(data_length/2)]
    output[1:data_length:2] = data_transformed[data_length-1:int(data_length/2-1):-1]
    
    return(output)
    

def fixed_point(t,N,I,a2):
    """
    Fixed Point function in support of py_kde.
    
    From Betov code:
    this implements the function t-zeta*gamma^[l](t)
 
    """
    l = 7
    f = 2 * (np.pi**(2*l)) * np.sum((I**l)*a2*np.exp(-I*(np.pi**2)*t))
    
    for s in np.arange(l-1,1,-1):
        K0 = np.prod(np.arange(1,2*s,2))/np.sqrt(2*np.pi)
        const = (1+(1/2)**(s+1/2))/3
        time = (2*const*K0/N/f)**(2/(3+2*s))
        f = 2*np.pi**(2*s)*np.sum(I**s*a2*np.exp(-I*(np.pi**2)*time))
    
    out = t - (2*N*np.sqrt(np.pi)*f)**(-2/5)
    return(abs(out))

def botev_r(data):
    """
    Calculate Botev bandwidth using modified version of R script from
    Botev et al., 2010.
    """
    path = os.path.join(dz_dir,'botev_mod.R')

    r = ro.r
    r.source(path)

    if isinstance(data,pd.Series):
        data.dropna(how='any',inplace=True)
        data = data.to_numpy()

    rdata = ro.vectors.FloatVector(data)
    
    rbandwidth = r.kde(rdata)
    
    bandwidth = np.asarray(rbandwidth)[0]
    
    return(bandwidth)

def vermeesch_r(data):
    """
    Calculate Botev bandwidth using R script from
    IsoplotR
    """
    path = os.path.join(dz_dir,'vermeesch_botev.R')

    r = ro.r
    r.source(path)

    if isinstance(data,pd.Series):
        data.dropna(how='any',inplace=True)
        data = data.to_numpy()
    
    rdata = ro.vectors.FloatVector(data)
    
    rbandwidth = r.botev(rdata)
    
    bandwidth = np.asarray(rbandwidth)[0]
    
    return(bandwidth)
        
    