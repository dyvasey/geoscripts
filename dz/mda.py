"""
Algorithims for automatic MDA calculation
"""
import numpy as np
import pandas as pd

from itertools import combinations

def weighted_mean(ages,errors,err_lev='2sig'):
    """
    Calculate weighted mean given ages and errors
    """
    weights = 1/errors**2
    
    wmean = np.average(ages,weights=weights)
    werror = 1/np.sqrt(np.sum(weights))
    
    deg_free = len(ages)-1
    
    if err_lev=='2sig':
        errors_1sig = errors/2
    
    elif err_lev =='1sig':
        errors_1sig = errors
    
    else:
        raise('Error level not valid')
    
    squares_summed = np.sum(((ages-wmean)/errors_1sig)**2)
    
    mswd = squares_summed/deg_free
    
    return(wmean,werror,mswd)
    
    
def ygc2sig(ages,err_lev='2sig'):
    """
    Youngest 3+ grains at 2 sigma overlap.
    
    Ages must be sorted and include errors as a Pandas dataframe
    """
    all_overlap = False
    force = False
    add_grain = False
    
    # Start with youngest 3 grains
    ngrains = 3
    grains = np.arange(0,3,1)
    
    
    # Calculate a new MDA if youngest ages don't overlap or adding new grains
    while (all_overlap==False)|(add_grain==True):
        # Attempt to get grains indicated
        try:
            ages_only = ages.loc[grains,'Best Age']
        
        # If grain not found, either force 3 youngest grains or stop adding grains
        except KeyError:
            if ngrains==3:
                print('No 3 ages overlap at 2 sigma. Defaulting to 3 youngest ages')
                grains = np.arange(0,3,1)
                force = True
            elif ngrains>3:
                ngrains = ngrains-1
                force = True
        
        print(ages_only)
        errors_only = ages.loc[grains,err_lev]
        
        # Calculate weighted mean
        wmean,werror,mswd = weighted_mean(ages_only,errors_only,err_lev)
        
        # Make sure using 2 sigma errors
        if err_lev=='2sig':
            errors_2sig = errors_only
        
        elif err_lev =='1sig':
            errors_2sig = errors_only*2
        
        else:
            raise('Error level not valid')
        
        
        # Check for overlap
        ages_max = (ages_only + errors_2sig)
        ages_min = (ages_only - errors_2sig)
        
        # Set overlaps for all possibilities
        overlaps = combinations(grains,2)
        
        for x,y in overlaps:
            overlap = (
                (ages_min[x] <= ages_max[y]) & (ages_min[y] <= ages_max[x])
                )
            
            print(overlap)
            # If no overlap between any ages, break the loop and go to next
            # oldest grain
            if overlap==False:
                grains = grains+1
                break
        
        # If all overlap, change overlap condition
        if overlap==True:
            all_overlap=True
        
        # Don't attempt to add grain if force flag is on
        if force==True:
            add_grain==False
            
        if (overlap==True)
        
    return(wmean,werror,mswd,ages_only,errors_only)
    
