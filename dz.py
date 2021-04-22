"""
Module for processing and plotting detrital zircon data
"""
import pickle

import seaborn as sns
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd

class DZSample:
    """ Object to hold detrital zircon sample metadata and ages """
    
    # Define basic attributes
    def __init__(self,name,latlon=None,agedata=None,color=None,
                 reported_age=None):
        self.name = name
        self.latlon= latlon
        self.agedata = agedata
        self.color = color
        self.reported_age = reported_age
        
        return

    def calc_discordance(self,col_238,col_207,cutoff=20,reverse_cutoff=-5,
                         age_cutoff=400):
        """
        Calculate discordance of 238U/206Pb and 207Pb/206Pb ages.
        """
        discordance = (
            1-(self.agedata[col_238]/self.agedata[col_207]))*100
        self.agedata['Discordance'] = discordance
        
        # Run filter
        discard = ((self.agedata[col_238]>age_cutoff) &
            ((discordance>cutoff) | (discordance<reverse_cutoff))
            )
        self.agedata['Discard'] = discard
        
        return(discordance,discard)    

    def calc_bestage(self,col_238,col_207,age_cutoff=900,filter_disc=True,
                     disc_cutoff=20,reverse_cutoff=-5,disc_age_cutoff=400):
        """
        Determine best age from 238U/206Pb and 207Pb/206Pb ages
        """
        # Use 238 age where 238 age is less than given age
        self.bestage = self.agedata[col_238].where(
            self.agedata[col_238]<age_cutoff,
            self.agedata[col_207])
        
        # Run discordance filter
        if filter_disc == True:
            discordance,discard = self.calc_discordance(
                col_238,col_207,cutoff=disc_cutoff,
                reverse_cutoff=reverse_cutoff,age_cutoff=disc_age_cutoff
                                                   )
            self.bestage = self.bestage[~discard]  
        
        
        self.bestage.name = 'Best Age'
        
        return(self.bestage)
    
    def kde(self,ax=None,log_scale=True,add_n=True,xaxis=True,**kwargs):
        """
        Plot KDE using best age.
        """
        if ax == None:
            ax = plt.gca()
            
        sns.kdeplot(self.bestage,log_scale=log_scale,label=self.name,
                    ax=ax,shade=True,color=self.color,**kwargs)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        if add_n == True:
            text = 'n = ' + str(self.bestage.count())
            ax.text(0.02,0.5,text,transform=ax.transAxes,fontweight='bold')
        
        if xaxis == False:
            ax.get_xaxis().set_visible(False)
        
        return(ax)
    
    def map_location(self,ax=None,crs=ccrs.PlateCarree(),**kwargs):
        """
        Add sample location to map
        """
        if ax == None:
            ax = plt.gca()
        
        # Plot according to one location or list of locations
        if isinstance(self.latlon,tuple):
            ax.scatter(self.latlon[1],self.latlon[0],transform=crs,
                   label=self.name,color=self.color,**kwargs)
        elif isinstance(self.latlon,list):
            lat = [x[0] for x in self.latlon]
            lon = [x[1] for x in self.latlon]
            ax.scatter(lon,lat,transform=crs,
                   label=self.name,color=self.color,**kwargs)
        
        return(ax)
    
    def export_ages(self,filename=None):
        if filename==None:
            filename = self.name + '.csv'
        
        self.bestage.to_csv(filename)
        
        return
    
    def save(self,filename=None):
        if filename==None:
            filename = self.name + '.dz'
        
        pickle.dump(self, open(filename,"wb"))
        
        return

def composite(samples,name,color=None):
    """
    Create composite DZ data from multiple samples
    """
    comp = DZSample(name,color=color)
    comp.bestage = pd.Series()
    comp.latlon = []
    comp.reported_age = []
    
    for sample in samples:
        comp.bestage = comp.bestage.append(sample.bestage)
        comp.latlon.append(sample.latlon)
        comp.bestage.name = 'Age (Ma)'
        comp.reported_age.append(sample.reported_age)
    
    return(comp)

def load(filename):
    
    dz = pickle.load(open(filename,"rb"))
    
    return(dz)

        
            
            
        