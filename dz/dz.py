"""
Module for processing and plotting detrital zircon data
"""
import pickle
import os

import seaborn as sns
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm

from matplotlib.colors import cnames
from matplotlib import cm

from geoscripts.dz import mda

class DZSample:
    """ Object to hold detrital zircon sample metadata and ages. """
    
    # Define basic attributes
    def __init__(self,name,latlon=None,agedata=None,color=None,
                 reported_age=None,source=None):
        """
        Constructs necessary attributes for DZ object.
        
        Parameters:
            name: Sample name
            latlon (tuple): Sample coordinates in latitude/longitude
            agedata: Raw age data for the sample
            color: Sample color for plotting
            reported_age: Depositional age for the sample
            source: Publication source for the sample
            
        Returns:
            None
        """
        self.name = name
        self.latlon= latlon
        self.agedata = agedata
        self.color = color
        self.reported_age = reported_age
        self.source = source
        
        return

    def calc_discordance(self,col_238,col_207,cutoff=10,reverse_cutoff=-10,
                         age_cutoff=400):
        """
        Calculate discordance of 238U/206Pb and 207Pb/206Pb ages.
        
        Defaults follow values typically used at Arizona LaserChron Center.
        
        Parameters:
            col_238: Series or array with 238U/206Pb ages
            col_207: Series or array with 207Pb/206Pb ages.
            cutoff: % discordance filter cutoff
            reverse_cutoff: % reverse discordance filter cutoff (negative)
            age_cutoff: Age (Ma) above which to run the discordance filter
        
        Returns:
            discordance: % discordance for each analysis
            discard: Boolean of whether age should be discarded for discordance
        """
        discordance = (
            1-(self.agedata[col_238]/self.agedata[col_207]))*100
        self.agedata.loc[:,'Discordance'] = discordance
        
        # Run filter
        discard = ((self.agedata[col_238]>age_cutoff) &
            ((discordance>cutoff) | (discordance<reverse_cutoff))
            )
        self.agedata.loc[:,'Discard'] = discard
        
        return(discordance,discard)    

    def calc_bestage(self,col_238,col_207,err_238=None,err_207=None,
                     age_cutoff=900,filter_disc=True,use_err=False,err_lev='2sig',
                     disc_cutoff=10,reverse_cutoff=-10,disc_age_cutoff=400):
        """
        Determine best age from 238U/206Pb and 207Pb/206Pb ages.
        
        Uses discordance filter from calc_discordance and default values from
        Arizona LaserChron Center.
        
        Parameters:
            col_238: Series or array with 238U/206Pb ages
            col_207: Series or array with 207Pb/206Pb ages.
            age_cutoff: Age above which to use 207Pb/206Pb age as best age
            filter_disc: Boolean for whether to use discordance filter
            disc_cutoff: % discordance cutoff for discordance filter
            reverse_cutoff: % reverse discordance cutoff for discordance filter
                            (negative)
            disc_age_cutoff: Age cutoff for discordance filter
            
        Returns:
            bestage: Best ages (Ma)
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
        
        if use_err == True:
            self.besterror = self.agedata[err_238].where(
                self.agedata[col_238]<age_cutoff,
                self.agedata[err_207])
            
            if filter_disc == True:
                self.besterror = self.besterror[~discard]
            self.error_level = err_lev
            self.besterror.name = err_lev
        
        
        return(self.bestage)
    
    def kde(self,ax=None,log_scale=True,add_n=True,xaxis=True,rug=True,
            method=None,ticks=[100,200,300,400,500,1000,2000,3000],
            **kwargs):
        """
        Plot KDE via Seaborn using best age.
        
        Parameters:
            ax: Axes on which to plot KDE
            log_scale: Whether to plot age on logarithmic scale
            add_n: Whether to add number of analyses to plot
            xaxis: Whether to show x axis labels
            rug: Whether to add ticks to bottom of plot
            method: Method for getting KDE bandwidth
        
        Returns:
            ax: Axes which KDE plotted
        """
        if ax == None:
            ax = plt.gca()
        
        # Transform data to log scale if needed for bandwidth calculation
        if log_scale==True:
            data_bw = np.log10(self.bestage)
        else:
            data_bw = self.bestage
        
        # STD of sample - transformed to log scale if needed - needed to
        # feed bandwidth factor to Seaborn
        std = data_bw.std()
        
        # Botev R script
        if method=='botev_r':
            from geoscripts.dz import botev
            bandwidth = botev.botev_r(data_bw)
            bw_method = bandwidth/std
        
        # Botev Python script - currently doesn't work
        elif method=='botev_py':
            print('Warning: Method may be unstable.')
            grid,density,bandwidth = botev.py_kde(data_bw)
            bw_method = bandwidth/std
            
        elif method=='vermeesch':
            from geoscripts.dz import botev
            bandwidth = botev.vermeesch_r(data_bw)
            bw_method = bandwidth/std
            
        # Use Seaborn default
        else:
            bw_method = 'scott'
            
            
        sns.kdeplot(self.bestage,log_scale=log_scale,label=self.name,
                    ax=ax,shade=True,color=self.color,gridsize=1000,
                    bw_method=bw_method,**kwargs)
        if rug == True:
            sns.rugplot(self.bestage,ax=ax,height=-0.03,clip_on=False,
                        color=self.color,expand_margins=False,
                        linewidth=1)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)
        
        if add_n == True:
            text = 'n = ' + str(self.bestage.count())
            ax.text(0.02,0.5,text,transform=ax.transAxes,fontweight='bold')
        
        if xaxis == False:
            ax.get_xaxis().set_visible(False)
        
        return(ax)
    
    def plot_agehf(self,hf_col,ax=None,**kwargs):
        if ax == None:
            ax = plt.gca()
        
        hf = self.agedata[hf_col].dropna(how='any')
        ages = self.bestage[hf.index]
        ax.scatter(ages,hf,**kwargs)
        
        return(ax)
    
    def kde_hf(self,hf_col,ax=None,include_ages=True,cmap='viridis',method=None,
               marker_color='red',xlim='auto',ylim='auto',**kwargs):
        
        if ax is None:
            ax=plt.gca()

        hf = self.agedata[hf_col].dropna(how='any')
        ages = self.bestage[hf.index]

        if xlim=='auto':
            xlim = (np.min(ages),np.max(ages))
        if ylim=='auto':
            ylim = (np.min(hf),np.max(hf))

        if method=='vermeesch':
            bw_ages = botev.vermeesch_r(ages)
            # bw_method_ages = bw_ages/ages.std()
            print(bw_ages)

            bw_hf = botev.vermeesch_r(hf)
            # bw_method_hf = bw_hf/hf.std()
            print(bw_hf)

            bw = [bw_ages,bw_hf]

        elif method=='botev_r':
            bw_ages = botev.botev_r(ages)
            # bw_method_ages = bw_ages/ages.std()
            print(bw_ages)

            bw_hf = botev.botev_r(hf)
            # bw_method_hf = bw_hf/hf.std()
            print(bw_hf)

            bw = [bw_ages,bw_hf]
        
        elif isinstance(method,list):
            bw=method
        
        else:
            bw = 'normal reference'

        # Get data into nx2 matrix
        
        data = np.vstack([ages,hf]).T

        # Get multivariate estimator from calculated bandwidth
        kde = sm.nonparametric.KDEMultivariate(data=data, var_type='cc', bw=bw)
        
        # Define grid and estimate density throughout grid
        xgrid = np.linspace(xlim[0],xlim[1],200)
        ygrid = np.linspace(ylim[0],ylim[1],200)
        
        xx,yy = np.meshgrid(xgrid,ygrid)

        # Need to convert meshgrid output into array of coordinates (mx2 where m is number of points)
        coords = np.append(xx.reshape(-1,1), yy.reshape(-1,1),axis=1)

        # Evaluate density at each coordinate
        z_coord = kde.pdf(coords)

        # Reshape z for use with imshow
        z = z_coord.reshape(len(ygrid), len(xgrid))

        #Normalize values
        z_norm = z/np.max(z)

        #ax.contourf(xgrid,ygrid,z,cmap=cmap,**kwargs)
        cmap = cm.get_cmap(cmap).copy()
        cmap.set_under(color='white',alpha=0)

        ax.imshow(z_norm, cmap=cmap,extent=[xlim[0],xlim[1],ylim[0],ylim[1]],aspect='auto',origin='lower',**kwargs)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axhline(0,color='black')
        ax.text(0.7,0.55,'CHUR',transform=ax.transAxes,fontsize=8)

        ax.set_xlabel('Age (Ma)')
        ax.set_ylabel('$\epsilon Hf_i$')

        if include_ages==True:
            self.plot_agehf(hf_col,ax=ax,facecolors=marker_color,edgecolors='black',label=self.name,s=4,
            linewidths=0.5)
        
        return(ax)
    
    def kde_img(self,log_scale=True,add_n=True,method=None,xlim=(10,4000),
                **kwargs):
        """
        Save KDE as image file tied to dz object.
        
        Parameters:
            log_scale: Whether to plot age on logarithmic scale
            add_n: Whether to add number of analyses to plot
            bw_adjust: Bandwidth adjustment via Seaborn
            xlim: Range of x axis
        
        Returns:
            None
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(xlim)
        
        self.kde(ax=ax,log_scale=log_scale,add_n=add_n,method=method,
                 **kwargs)
        
        #sns.kdeplot(self.bestage,log_scale=log_scale,label=self.name,
        #            ax=ax,shade=True,color=self.color,bw_adjust=bw_adjust,
        #            **kwargs).set(title=self.name)
        
        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        #ax.spines['left'].set_visible(False)
        #ax.get_yaxis().set_visible(False)
        
        #if add_n == True:
        #    text = 'n = ' + str(self.bestage.count())
        #    ax.text(0.02,0.5,text,transform=ax.transAxes,fontweight='bold')
        
        path = 'dz/'
        os.makedirs(path,exist_ok=True)
        name = self.name+'_KDE.png'
        self.kde_path = path+name
        fig.savefig(path+name)
        
        return
    
    def cad(self,ax=None,depage=None,**kwargs):
        """
        Plot cumulative age distribution
        """
        if ax == None:
            ax = plt.gca()
        
        if depage == None:
            sns.ecdfplot(self.bestage,ax=ax,label=self.name,
                         **kwargs)
            
        elif depage == 'youngest':
            norm_ages = self.bestage - np.min(self.bestage)
            sns.ecdfplot(norm_ages,ax=ax,label=self.name,
                         **kwargs)
        
        else:
            norm_ages = self.bestage - depage
            sns.ecdfplot(norm_ages,ax=ax,label=self.name,
                         **kwargs)
        
        return(ax)
    
    def cawood_classify(self,depage='youngest',plot=False):
        """
        Classify sample using Cawood et al., 2012 framework.
        """
        
        if depage == 'youngest':
            depage = np.min(self.bestage)
        
        norm_ages = self.bestage - depage
        
        ages_sorted = np.sort(norm_ages)
        
        cumden = np.arange(len(ages_sorted))/len(ages_sorted)
        
        # Get age where 5% density reached
        step1_index = np.min(np.where(cumden>=0.05))
        step1_age = ages_sorted[step1_index]
        
        if step1_age >= 150:
            self.cawood = 'extensional'
            
            text = self.name + '\n' + str(step1_age) + '\n' + self.cawood
        
        # Get age where 30% density reached
        elif step1_age < 150:
            step2_index = np.min(np.where(cumden>=0.3))
            step2_age = ages_sorted[step2_index]
            
            if step2_age >= 100:
                self.cawood = 'collisional'
            
            elif step2_age < 100:
                self.cawood = 'convergent'
                
            else:
                raise('Something went wrong')
            
            text = (
                self.name + '\n' + str(step1_age) + ', ' + 
                str(step2_age) + '\n' + self.cawood
                )
            
        else:
            raise('Something went wrong')
        
        if plot:
            color_dict = {'convergent':'red','collisional':'blue',
                          'extensional':'green'}
            
            fig,ax = plt.subplots(1)
            ax.step(ages_sorted,cumden, color=color_dict[self.cawood])
            ax.annotate(text,xy=(500,0.2))
            
            ax.axhline(0.05,color='blue')
            ax.axhline(0.3,color='red')
            ax.axvline(150,color='blue')
            ax.axvline(100,color='red')    
            
            ax.set_xlim(0,3000)
            ax.set_ylim(0,1)
        return
    
    def calc_mda(self,method='ygc2sig',grains=None,plot=True):
        """
        Calculate and plot maximum depositional age using selected grains
        """
        age_errors = pd.concat([self.bestage,self.besterror],axis=1)
        ages_sorted = age_errors.sort_values(by=['Best Age'],ignore_index=True)
        err_lev = self.error_level
        
        if method=='ygc2sig':
            (self.mda,self.mda_err,self.mda_mswd,
             self.mda_ages,self.mda_errors,self.mda_success) = mda.ygc2sig(ages_sorted,err_lev)
            if self.mda_success==False:
                print('3 grains at 2 sigma did not succeed. Trying 2 grains at 1 sigma')
                method = 'ygc1sig'
        
        if method=='ygc1sig':
            (self.mda,self.mda_err,self.mda_mswd,
            self.mda_ages,self.mda_errors,self.mda_success) = mda.ygc1sig(ages_sorted,err_lev)  
            
        if method=='manual':
            self.mda_ages = ages_sorted.loc[grains,'Best Age']
    
            self.mda_errors = ages_sorted.loc[grains,err_lev]
            
            weights = 1/self.mda_errors**2
            
            self.mda = np.average(self.mda_ages,weights=weights)
            self.mda_err = 1/np.sqrt(np.sum(weights))
            
            deg_free = len(grains)-1
            
            if err_lev=='2sig':
                errors_1sig = self.mda_errors/2
            
            elif err_lev =='1sig':
                errors_1sig = self.mda_errors
            
            else:
                raise('Error level not valid')
            
            squares_summed = np.sum(((self.mda_ages-self.mda)/errors_1sig)**2)
            
            self.mda_mswd = squares_summed/deg_free
        
        if plot==True:
            fig,ax = plt.subplots(1,dpi=300)
            mda.plot_weighted_mean(self.mda_ages,self.mda_errors,self.mda,
                                   self.mda_err,self.mda_mswd,err_lev=err_lev,
                                   ax=ax,label=self.name)
        
        return(self.mda,self.mda_err,self.mda_ages,self.mda_errors)
        
    def map_location(self,ax=None,crs=ccrs.PlateCarree(),**kwargs):
        """
        Add sample location to map with Cartopy.
        
        Parameters:
            ax: Axes on which to plot location
            crs: Cartopy coordinate reference system
        
        Returns:
            ax: Axes with location plotted
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
        """
        Export best ages to CSV file for external processing.
        
        Parameters:
            filename: name of CSV file (optional)
        
        Returns:
            None
        """
        path = 'dz/'
        os.makedirs(path,exist_ok=True)
        
        if filename==None:
            filename = self.name + '.csv'
        
        self.bestage.to_csv(path+filename)
        
        return
    
    def save(self,filename=None):
        """
        Save DZ object to .dz file to reload in other scripts.
        
        Parameters:
            filename: name of file (optional)
        """
        path = 'dz/'
        os.makedirs(path,exist_ok=True)
        
        if filename==None:
            filename = self.name + '.dz'
        
        pickle.dump(self, open(path+filename,"wb"))
        
        return

def composite(samples,name,color=None):
    """
    Create composite DZ data from multiple samples.
    
    Parameters:
        samples: List of DZ objects
        name: Name of composite data
        color: Color to use for plotting
        
    Returns:
        comp: DZ object with composite data
    
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

    comp.bestage = comp.bestage.reset_index(drop=True)
    
    return(comp)

def load(filename,path='dz/'):
    """
    Load .dz file into DZ object.
    
    Parameters:
        filename: Name of .dz in file in dz/ path.
        
    Returns:
        dz: DZ object with loaded data.
    """
    dz = pickle.load(open(path+filename,"rb"))
    
    return(dz)

def write_file(samples,filename):
    """
    Create point shapefile from multiple samples.
    
    Parameters:
        samples: List of DZ objects
        filename: Name of shapefile
    
    Returns:
        gdf: GeoPandas GeoDataFrame written to the shapefile
    """
    latitude = []
    longitude = []
    name = []
    reported_age = []
    kde_path = []
    source = []
    color = []
    
    for sample in samples:
        latitude.append(sample.latlon[0])
        longitude.append(sample.latlon[1])
        name.append(sample.name)
        reported_age.append(sample.reported_age)
        kde_path.append(sample.kde_path)    
        source.append(sample.source)
        
        color_hex = cnames[sample.color]
        color.append(color_hex)
    
    geometry = gpd.points_from_xy(longitude,latitude)
    data = {'name':name,'reported_age':reported_age,
            'kde_path':kde_path,'source':source,'color':color}
    gdf = gpd.GeoDataFrame(data,geometry=geometry)
    
    gdf.to_file(filename,crs='EPSG:4326')
    return(gdf)

def load_all(path='dz/'):
    """
    Load all dz files in directory using the load function
    
    Parameters:
        path: Path to directory with .dz files
    
    Returns:
        samples: List of loaded DZ objects
    """
    samples = []
    for file in os.listdir(path):
        if file.endswith('.dz'):
            obj = load(file,path=path)
            samples.append(obj)
    
    return(samples)
            

            
        