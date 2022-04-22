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
    success=True
    
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
            # Grains reset to default and forced to be accepted.
            if ngrains==3:
                print('No 3 ages overlap at 2 sigma. Defaulting to 3 youngest ages')
                grains = np.arange(0,3,1)
                force = True
            # Setting add grain to false should break while loop
            elif ngrains>3:
                add_grain = False
        
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
            

            if (overlap==False):
            # If no overlap between any ages at 3 grains, break the loop and go to next
            # oldest grain
                if ngrains==3:
                    grains = grains+1
                    break
                # If trying more than 3 grains, revert back to 1 fewer grain
                # and take previous age.
                if ngrains>3:
                    ngrains = ngrains-1
                    
                    ages_only = ages_only[:-1]
                    errors_only = errors_only[:-1]
                    
                    wmean,werror,mswd = weighted_mean(
                        ages_only,errors_only,err_lev)
                    
                    # Should break while loop
                    add_grain=False
                    break
        
        # If all overlap, change overlap condition and try adding grain
        if overlap==True:
            all_overlap=True
            add_grain=True
            ngrains = ngrains+1
            grains = np.arange(grains[0],grains[-1]+2,1)
        
        # Break the loop here if the force flag is on
        if force==True:
            success=False
            break
        
        
    return(wmean,werror,mswd,ages_only,errors_only,success)

def ygc1sig(ages,err_lev='2sig'):
    """
    Youngest 2+ grains at 1 sigma overlap.
    
    Ages must be sorted and include errors as a Pandas dataframe
    """
    all_overlap = False
    force = False
    add_grain = False
    
    # Start with youngest 2 grains
    ngrains = 2
    grains = np.arange(0,2,1)
    
    
    # Calculate a new MDA if youngest ages don't overlap or adding new grains
    while (all_overlap==False)|(add_grain==True):

        # Attempt to get grains indicated
        try:
            ages_only = ages.loc[grains,'Best Age']
        
        # If grain not found, either force 3 youngest grains or stop adding grains
        except KeyError:
            # Grains reset to default and forced to be accepted.
            if ngrains==2:
                print('No 2 ages overlap at 1 sigma. Defaulting to 2 youngest ages')
                grains = np.arange(0,2,1)
                force = True
            # Setting add grain to false should break while loop
            elif ngrains>2:
                add_grain = False
        
        errors_only = ages.loc[grains,err_lev]
        
        # Calculate weighted mean
        wmean,werror,mswd = weighted_mean(ages_only,errors_only,err_lev)
        
        # Make sure using 1 sigma errors
        if err_lev=='2sig':
            errors_1sig = errors_only/2
        
        elif err_lev =='1sig':
            errors_1sig = errors_only
        
        else:
            raise('Error level not valid')
        
        
        # Check for overlap
        ages_max = (ages_only + errors_1sig)
        ages_min = (ages_only - errors_1sig)
        
        # Set overlaps for all possibilities
        overlaps = combinations(grains,2)
        
        for x,y in overlaps:
            overlap = (
                (ages_min[x] <= ages_max[y]) & (ages_min[y] <= ages_max[x])
                )
            

            if (overlap==False):
            # If no overlap between any ages at 2 grains, break the loop and go to next
            # oldest grain
                if ngrains==2:
                    grains = grains+1
                    break
                # If trying more than 2 grains, revert back to 1 fewer grain
                # and take previous age.
                if ngrains>2:
                    ngrains = ngrains-1
                    
                    ages_only = ages_only[:-1]
                    errors_only = errors_only[:-1]
                    
                    wmean,werror,mswd = weighted_mean(
                        ages_only,errors_only,err_lev)
                    
                    # Should break while loop
                    add_grain=False
                    break
        
        # If all overlap, change overlap condition and try adding grain
        if overlap==True:
            all_overlap=True
            add_grain=True
            ngrains = ngrains+1
            grains = np.arange(grains[0],grains[-1]+2,1)
        
        # Break the loop here if the force flag is on
        if force==True:
            break
        
        
    return(wmean,werror,mswd,ages_only,errors_only)
    
