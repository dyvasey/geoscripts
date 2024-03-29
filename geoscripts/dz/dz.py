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
from matplotlib.patches import Ellipse
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

    def calc_discordance(self,col_238,col_207,cutoff=20,reverse_cutoff=-10,
                         age_cutoff=600):
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
                     disc_cutoff=20,reverse_cutoff=-10,disc_age_cutoff=600):
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
                    ax=ax,fill=True,color=self.color,gridsize=1000,
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
    
    def pie(self,bins,ax=None,**kwargs):
        """
        Plot pie chart showing relative percentages of zircon ages.

        Requires that bestage is assigned
        """

        if ax == None:
            ax = plt.gca()

        hist = np.histogram(self.bestage,bins=bins)

        labels = [str(bins[x])+ '-' + str(bins[x+1]) + ' Ma' for x in range(len(bins)-1)]

        ax.pie(hist[0],labels=labels,**kwargs)

        return(ax)

        

    def plot_agehf(self,hf_col,ax=None,**kwargs):
        """
        Plot zircon age against Hf isotopic value

        Parameters:
            hf_col: Column in agedata with Hf values
            ax: Axes on which to plot values
        
        Returns:
            ax: Axes with plot
        """
        if ax == None:
            ax = plt.gca()
        
        hf = self.agedata[hf_col].dropna(how='any')
        ages = self.bestage[hf.index]
        ax.scatter(ages,hf,**kwargs)
        
        return(ax)
    
    def kde_hf(self,hf_col,ax=None,include_ages=True,cmap='viridis',method=None,
               marker_color='red',xlim='auto',ylim='auto',**kwargs):
        """
        Plot bivariate KDE of age vs. Hf

        Parameters:
            hf_col: Column in agedata with Hf values
            ax: Axes on which to draw plot
            include_ages: Boolean for whether to include scatter plot of individual ages
            cmap: Colormap for KDE
            method: Method for bandwidth calculation. Options include 'vermeesch','botev_r', 
                a list of bandwidths for the x and y axes, or the 'normal reference' 
                bandwidth in statsmodels. Current recommended use is a list of
                bandwidths.
            marker_color: Color for scatter plot of ages if used.
            xlim: Limits of x axis. Default is automatic determination of minimum and
                maximum values, or can be provided as a tuple.
            ylim: Limits of y axis. Default is automatic determination of minimum and
                maximum values, or can be provided as a tuple.
        """
        if ax is None:
            ax=plt.gca()

        hf = self.agedata[hf_col].dropna(how='any')
        ages = self.bestage[hf.index]

        if xlim=='auto':
            xlim = (np.min(ages),np.max(ages))
        if ylim=='auto':
            ylim = (np.min(hf),np.max(hf))

        if method=='vermeesch':
            import botev
            bw_ages = botev.vermeesch_r(ages)
            print(bw_ages)

            bw_hf = botev.vermeesch_r(hf)
            print(bw_hf)

            bw = [bw_ages,bw_hf]

        elif method=='botev_r':
            import botev
            bw_ages = botev.botev_r(ages)
            print(bw_ages)

            bw_hf = botev.botev_r(hf)
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
        
        path = 'dz/'
        os.makedirs(path,exist_ok=True)
        name = self.name+'_KDE.png'
        self.kde_path = path+name
        fig.savefig(path+name)
        
        return
    
    def cad(self,ax=None,depage=None,**kwargs):
        """
        Plot cumulative age distribution (CAD).

        Parameters:
            ax: Axes on which to plot CAD
            depage: Age to use as depositional age. If specified,
                subtracts each age by this value, or by the 
                youngest age if 'youngest.'

        Returns:
            ax: Axes with plot
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

        Parameters:
            depage: Age to use as depositional age. If specified,
                subtracts each age by this value, or by the 
                youngest age if 'youngest.'
            plot: Boolean for whether to plot CAD colored
                by Cawood classification.
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
    
    def calc_mda(self,method='ygc2sig',grains=None,plot=True,overdisperse=False,systematic=False,
                 filter235238=False,cutoffs235238=(80,110)):
        """
        Calculate and plot maximum depositional age (MDA).

        Requires that bestage has already been determined.

        Parameters:
            method: Method to use for MDA. Current options are 'ygc2sig','ygc1sig', and 'manual.'
            grains: Indicies of grains to use if using 'manual' method.
            plot: Boolean of whether to output MDA plot.
            overdisperse: Whether to factor in overdispersion in final error for MDA.
            systematic: Whether to factor in systematic error in final error for MDA.
        
        Returns:
            self.mda: Maximum depositional age (Ma)
            self.mda_err: Error on MDA (Ma)
            self.mda_ages: Ages used in MDA calculation (Ma)
            self.mda_errors: Errors for ages used in MDA calculation (Ma)
        """
        age_errors = pd.concat([self.bestage,self.besterror],axis=1)

        if filter235238:
            concordance = np.round((self.age_238/self.age_235)*100,0)
            accept = ((concordance >= cutoffs235238[0]) & (concordance <= cutoffs235238[1]))
            age_errors = age_errors[accept]
            print(np.min(concordance),np.max(concordance))
        
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

        if systematic==True:
            print('Propagating systematic error - ',self.syst_238,'\nOriginal error: ',self.mda_err)
            # Convert % error to absolute error
            syst_err = self.syst_238/100 * self.mda
            self.mda_err = np.sqrt(self.mda_err**2 + syst_err**2)
        
        else:
            self.syst_238 = None

        if overdisperse==True:
            print('Using overdispersion factor, original error: ',self.mda_err)
            self.mda_err = self.mda_err*np.sqrt(self.mda_mswd)
        
        if plot==True:
            fig,ax = plt.subplots(1,dpi=300)
            mda.plot_weighted_mean(self.mda_ages,self.mda_errors,self.mda,
                                   self.mda_err,self.mda_mswd,err_lev=err_lev,
                                   ax=ax,label=self.name,syst_error=systematic,
                                   syst_238=self.syst_238)
        
        return(self.mda,self.mda_err,self.mda_ages,self.mda_errors)

    def calc_ysg(self,systematic=True,filter235238=False,cutoffs235238=(80,110)):

        # Sort by best age
        age_errors = pd.concat([self.bestage,self.besterror],axis=1).dropna(how='any')

        # Filter for concordance
        if filter235238:
            concordance = np.round((self.age_238/self.age_235)*100,0)
            accept = ((concordance >= cutoffs235238[0]) & (concordance <= cutoffs235238[1]))
            age_errors = age_errors[accept]

        ages_sorted = age_errors.sort_values(by=['Best Age'],ignore_index=True)

        # Find youngest age
        self.ysg = ages_sorted.iloc[0,0]

        # Propagate systematic error
        ysg_syst = (self.syst_238/100)*self.ysg 
        self.ysg_err = np.sqrt(ages_sorted.iloc[0,1]**2 + ysg_syst**2)

        return(self.ysg,self.ysg_err)

    def convert_1sigto2sig(self):
        """
        Convert 1 sigma errors on best ages to 2 sigma.

        Parameters:
            None
        
        Returns:
            None
        """

        if self.error_level=='2sig':
            print('Errors already at 2sig')
        elif self.error_level=='1sig':
            print('Converting...')
            self.besterror = self.besterror*2
            self.error_level='2sig'
            self.besterror.name = self.error_level
        else:
            print('Something went wrong')
        return
    
    def plot_mda(self,ax=None,syst_error=False):
        """
        Plot sample MDA as weighted mean plot.

        Parameters:
            ax: Axes on which to plot.
            syst_error: Boolean for whether to include systematic error
        
        Returns:
            ax: Axes with plot.
        """

        if syst_error==False:
            mda.plot_weighted_mean(self.mda_ages,self.mda_errors,self.mda,
                            self.mda_err,self.mda_mswd,err_lev=self.error_level,
                            ax=ax,label=self.name,syst_error=syst_error,syst_238=None)
        
        elif syst_error==True:
            mda.plot_weighted_mean(self.mda_ages,self.mda_errors,self.mda,
                self.mda_err,self.mda_mswd,err_lev=self.error_level,
                ax=ax,label=self.name,syst_error=syst_error,syst_238=self.syst_238)
    
        return(ax)

    def plot_concordia(self,col_238,err_238,col_207,err_207,ax=None,grains='all',
                       err_lev='2sig',mda=False,inverted_ratios=False,percent_error=True):

        """
        Errors handled incorrectly. Function currently not useable.
        """

        if ax is None:
            ax=plt.gca()
        
        # Plot concordia line
        ages_phan1 = np.arange(0,300,50)[1:]
        ages_phan2 = np.arange(300,1500,200)
        ages_prec = np.arange(1500,4000,500)
        ages_all = np.hstack([ages_phan1,ages_phan2,ages_prec])

        ages_curve = np.arange(0,4000,10)[1:]
        
        ratios = age2ratios(ages_all)
        ratios_curve = age2ratios(ages_curve)

        ax.plot(ratios_curve[0],ratios_curve[1],color='red')
        ax.scatter(ratios[0],ratios[1],color='red')

        for k,v in enumerate(ages_all):
            ax.annotate(v,(ratios[0][k],ratios[1][k]))

        # Plot data
        if mda:
            grain_min = np.min(self.mda_ages.index)
            grain_max = np.max(self.mda_ages.index)

        elif grains=='all':
            grain_min = 0
            grain_max = len(self.agedata[col_238])

        else:
            grain_min = np.min(grains)
            grain_max = np.max(grains)

        org_data = pd.concat([self.agedata[col_238],self.agedata[err_238],
                               self.agedata[col_207],self.agedata[err_207]],axis=1)

        org_data.columns = ['U238Pb206','err238206','Pb207Pb206','err207206']

        if inverted_ratios:
            org_data['U238Pb206'] = 1/org_data['U238Pb206']
            org_data['Pb207Pb206'] = 1/org_data['Pb207Pb206']
        
        # Convert percentage errors to absolute
        if percent_error:
            org_data['err238206'] = (org_data['err238206']/100)*org_data['U238Pb206'] 
            org_data['err207206'] = (org_data['err207206']/100)*org_data['Pb207Pb206'] 

        data_sorted = org_data.sort_values(by=['U238Pb206'],ascending=False,
                                           ignore_index=True)

        self.U238Pb206 = data_sorted.loc[grain_min:grain_max,'U238Pb206']
        self.Pb207Pb206 = data_sorted.loc[grain_min:grain_max,'Pb207Pb206']

        if err_lev=='2sig':
            self.err238206 = data_sorted.loc[grain_min:grain_max,'err238206']
            self.err207206 = data_sorted.loc[grain_min:grain_max,'err207206']

        elif err_lev=='1sig':
            self.err238206 = data_sorted.loc[grain_min:grain_max,'err238206']*2
            self.err207206 = data_sorted.loc.loc[grain_min:grain_max,'err207206']*2

        xlims = (np.min(self.U238Pb206-self.err238206)-2,np.max(self.U238Pb206+self.err238206)+2)
        ylims = (np.min(self.Pb207Pb206-self.err207206)-0.01,np.max(self.Pb207Pb206+self.err207206)+0.01)

        ax.set_xlim(0,xlims[1])
        ax.set_ylim(0.045,ylims[1])

        for k,v in enumerate(self.U238Pb206):
            e = Ellipse((self.U238Pb206.iloc[k],self.Pb207Pb206.iloc[k]),
                        width=self.err238206.iloc[k]*2,
                        height=self.err207206.iloc[k]*2,
                        linewidth=1,edgecolor='black',
                        alpha=0.5)
        
            ax.add_patch(e)

        return(ax)

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
    
    def save(self,filename=None,path='dz/'):
        """
        Save DZ object to .dz file to reload in other scripts.
        
        Parameters:
            filename: name of file (optional)

        Returns:
            None
        """
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
    comp.bestage = pd.Series(dtype='float64')
    comp.latlon = []
    comp.reported_age = []
    
    for sample in samples:
        comp.bestage = pd.concat([comp.bestage,sample.bestage])
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

def age2ratios(age,lambda238=1.55125e-10,lambda235=9.8485e-10):
    age_yr = age*1e6
    ratio238206 = 1/(np.exp(lambda238*age_yr)-1)
    ratio207206 = (
        (1/137.818)*(np.exp(lambda235*age_yr)-1)/(np.exp(lambda238*age_yr)-1)
    )

    return(ratio238206,ratio207206)
            

            
        