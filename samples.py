"""
Module for handling and processing geologic sample data
"""
import pickle
import os
from re import search

import seaborn as sns
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

from geoscripts import latlon

class Sample:
    """ Object to hold geologic sample data """
    # Basic attributes
    def __init__(self,name,lat=None,lon=None,lithology=None,source=None):
        """
        Constructs necessary attributes for sample object.
        
        Parameters:
            name: Sample name
            lat: Sample latitude (DD)
            lon: Sample longitude (DD)
            lithology: Sample rock type
            source: Source of the sample
            
        Returns:
            None
        """
        self.name = name
        self.lat = lat
        self.lon = lon
        self.lithology = str(lithology)
        self.source = source
        
        return
    
    def save(self,filename=None):
        """
        Save sample object to .smp file to reload in other scripts.
        
        Parameters:
            filename: name of file (optional)
        """
        path = 'smp/'
        os.makedirs(path,exist_ok=True)
        
        if filename==None:
            filename = self.name + '.smp'
        
        pickle.dump(self, open(path+filename,"wb"))
        
        return
    
    def map_location(self,ax=None,crs=ccrs.PlateCarree(),
                     add_label=True,**kwargs):
        """
        Add sample location to map with Cartopy.
        
        Parameters:
            ax: Axes on which to plot location
            crs: Cartopy coordinate reference system
            add_label: Label to add to map
        
        Returns:
            ax: Axes with location plotted
        """
        if ax == None:
            ax = plt.gca()
        
        ax.scatter(self.lon,self.lat,label=self.name,transform=crs,**kwargs)
        
        if add_label is True:
            ax.annotate(self.name,xy=(self.lon,self.lat))
        
        return(ax)
    
    def dms2deg(self,latdeg,londeg,latmin,lonmin,latsec=0,lonsec=0):
        """
        Calculate latitude/longitude in decimal degrees from dms values.
    
        Parameters:
            latdeg: Latitude degrees
            londeg: Longitude degrees
            latmin: Latitude minutes
            lonmin: Longitude minutes
            latsec: Latitude seconds
            lonsec: Longitude seconds
        
        Returns:
            None
        """
        lat,lon = latlon.dms2deg(latdeg,londeg,latmin,lonmin,latsec,lonsec)
        
        self.lat = lat
        self.lon = lon
        
        return
    
    def UTM2deg(self,easting,northing,zone,south=False):
        """
        Calculate latitude/longitude from UTM.
    
        Parameters:
            easting: UTM easting (m)
            northing: UTM northing (m)
            zone: UTM zone as integer
            south: Whether UTM zone is from the southern hemisphere
        """
        lat,lon = latlon.UTM2latlon(easting,northing,zone,south)
        
        self.lat = lat
        self.lon = lon
        
        return
    
    def standardize_lith(self):
        """
        Attempt to standardize lithology names
        """
        sandstone = ['ss','Ss','s/s','S/s','sandstone','Sandstone','wacke',
                     'Wacke']
        quartzite = ['quartzite','Quartzite']
        gneiss = ['gneiss','Gneiss']
        
        liths = [sandstone,quartzite,gneiss]
        joins = ['|'.join(x) for x in liths]
        
        if (
                (search(joins[0],self.lithology)) and 
                (not search(joins[2],self.lithology))
                ):
            self.standard_lith = 'Sandstone'
        
        elif (
                (search(joins[1],self.lithology))
                ):
            self.standard_lith = 'Quartzite'
        
        elif (
                (search(joins[2],self.lithology))
                ):
            self.standard_lith = 'Gneiss'
        
        else:
            self.standard_lith = 'Undetermined'
        
        return
            
class ProcessedSample(Sample):
    """
    Sample with additional data on thin sections and mineral separations        
    """
    def __init__(self,name,lat,lon,lithology,source,
                 current_location=None,ts=False,separate=False,dz=None,
                 tchron=None,gchron=None):
        super().__init__(name,lat,lon,lithology,source)
        self.current_location = current_location
        self.ts = ts
        self.separate = separate
        self.dz = dz
        self.tchron = tchron
        self.gchron = gchron
    
def load(filename):
    """
    Load .smp file into Sample object.
    
    Parameters:
        filename: Name of .smp in file in smp/ path.
        
    Returns:
        smp: DZ object with loaded data.
    """
    path = 'smp/'
    smp = pickle.load(open(path+filename,"rb"))
    
    return(smp)

def load_all(path='smp/'):
    """
    Load all .smp files in directory using the load function
    
    Parameters:
        path: Path to directory with .smp files
    
    Returns:
        samples: List of loaded Sample objects
    """
    smps = []
    for file in os.listdir(path):
        if file.endswith('.smp'):
            obj = load(file)
            smps.append(obj)
    
    return(smps)

def make_gdf(smps,fields=['name','lat','lon','lithology','source']):
    """
    Convert list of Sample objects to GeoDataFrame
    """
    full_data = []
    for sample in smps:
        data = {}
        for field in fields:
            data[field] = getattr(sample,field)
        full_data.append(data)
    
    df = pd.DataFrame.from_records(full_data)
    
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon,df.lat))
    
    return(gdf)

def process_sample(sample,current_location=None,ts=False,separate=False,dz=None,
                 tchron=None,gchron=None):
    """
    Convert sample to processed sample
    """
    psample = ProcessedSample(sample.name,sample.lat,sample.lon,
                              sample.lithology,sample.source,current_location,
                              ts,separate,dz,tchron,gchron)

    return(psample)    
    
    