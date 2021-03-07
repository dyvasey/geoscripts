"""
Module for processing and plotting detrital zircon data
"""
import seaborn as sns
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

class DZSample:
    """ Object to hold detrital zircon sample metadata and ages """
    
    # Define basic attributes
    def __init__(self,name,latlon=None,agedata=None):
        self.name = name
        self.latlon= latlon
        self.agedata = agedata
        
        return

    def calc_bestage(self,col_238,col_207,cutoff=900):
        """
        Determine best age from 238U/206Pb and 207Pb/206Pb ages
        """
        
        # Use 238 age where 238 age is less than given age
        self.bestage = self.agedata[col_238].where(
            self.agedata[col_238]<cutoff,
            self.agedata[col_207])
        self.bestage.name = 'Best Age'
        
        return(self.bestage)
    
    def kde(self,ax=None,log_scale=True,**kwargs):
        """
        Plot KDE using best age.
        """
        if ax == None:
            ax = plt.gca()
            
        sns.kdeplot(self.bestage,log_scale=log_scale,label=self.name,
                    ax=ax,shade=True,**kwargs)
        
        return(ax)
    
    def map_location(self,ax=None,crs=ccrs.PlateCarree(),**kwargs):
        """
        Add sample location to map
        """
        if ax == None:
            ax = plt.gca()
        
        ax.scatter(self.latlon[1],self.latlon[0],transform=crs,
                   label=self.name,**kwargs)
        
        return(ax)
        