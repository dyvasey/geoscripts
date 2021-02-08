"""
Module for making geochemical plots.
"""
import string

import matplotlib.pyplot as plt
import pandas as pd
import pyrolite.plot
from mpltern.ternary.datasets import get_triangular_grid

def TAS(SiO2,Na2O,K2O,ax=None,first= [],**plt_kwargs):
    """
    Plots total alkali-silica (TAS) diagram after Le Bas et al., 1986. 
    
    Plot divided into alkaline and subalkaline fields after Irvine and
    Barangar, 1971. Values used for plot lines were taken from source code
    of GCDKit (Janousek et al., 2006).
    
    Parameters:
        SiO2: List of SiO2 values (wt. %)
        Na2O: List of Na2O values (wt. %)
        K2O: List of K2O values (wt. %)
        ax: Axes on which to plot the diagram
        first: Empty list by default. If empty, lines/labels will plot
    
    Returns:
        ax: Axes with TAS plotted
    """
    if ax is None:
        ax = plt.gca()
    
    # Calculate total alkalis
    alkalis = Na2O + K2O
    
    # Plot data
    ax.scatter(SiO2,alkalis, **plt_kwargs)
    
    # Check if first empty to avoid repeat plotting of TAS grid/labels
    if first == []:
    # Create lines
        line1 = [[30,41,41,45,48.4,52.5,30],[0,0,7,9.4,11.5,14,24.15]]
        line2 = [[41,45,45,41],[0,0,3,3]]
        line3 = [[45,52,52,45],[0,0,5,5]]
        line4 = [[52,57,57,52],[0,0,5.9,5]]
        line5 = [[57,63,63,57],[0,0,7,5.9]]
        line6 = [[63,77,69,63],[0,0,8,7]]
        line7 = [[77,100,100,69,69],[0,0,25,25,8]]
        line8 = [[45,52,49.4],[5,5,7.3]]
        line9 = [[52,57,53,49.4],[5,5.9,9.3,7.3]]
        line10 = [[57,63,57.6,53],[5.9,7,11.7,9.3]]
        line11 = [[63,69,69,57.6],[7,8,17.73,11.7]]
        line12 = [[41,45,45,49.4,45,41],[3,3,5,7.3,9.4,7]]
        line13 = [[49.4,53,48.4,45],[7.3,9.3,11.5,9.4]]
        line14 = [[53,57.6,52.5,48.4],[9.3,11.7,14,11.5]]
        line15 = [[57.6,69,30],[11.7,17.73,24.15]]
        lines = [line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,
                 line11,line12,line13,line14,line15]
        
        # Create labels
        labelsx = [43,48.5,54.8,59.9,67,75,63.5,57.8,52.95,49.2,45,49.2,53,57,
                   43]
        labelsy = [1.55,2.8,3,3,3,8,11,8.5,7,5.65,7,9.3,11.5,14,12]
        labeltext = ['Picrobasalt','Basalt','Basaltic\nAndesite','Andesite',
                     'Dacite','Rhyolite','Trachyte/Trachydacite',
                     'Trachy-andesite','Basaltic-\ntrachy-andesite',
                     'Trachy-basalt','Tephrite/\nBasanite','Phono-tephrite',
                     'Tephri-phonolite','Phonolite','Foidite']
  
        # Create subalkaline/alkaline fields
        subalkx = [39.2,40,43.2,45,48,50,53.7,55,60,65,77.4]
        subalky = [0,0.4,2,2.8,4,4.75,6,6.4,8,8.8,10]
        
        
        # Plot Subalkaline/Alkaline line
        ax.plot(subalkx,subalky,'r--')
        ax.text(38,2,'Alkaline',rotation=45,color='r',ha='center',va='center')
        ax.text(49,2,'Subalkaline',rotation=45,color='r',ha='center',
                va='center')
        
        #Set axes limits
        ax.set_xlim(35,80)
        ax.set_ylim(0,16)
       
        for z in range(15): # Loop through and plot TAS lines
            ax.plot(lines[z][0],lines[z][1],'k')
            ax.text(labelsx[z],labelsy[z],labeltext[z],color='k',
                    ha='center',va='center',fontsize=10)
        
        # Avoid repeat grid plotting
        first.append('Not First')      
    return(ax)

def TASsm(SiO2,Na2O,K2O,ax=None,first= [],**plt_kwargs):
    """
    Plots small total alkali-silica (TAS) diagram after Le Bas et al., 1986. 
    
    This version plots a small TAS diagram with minimal text/labels in order
    to accomodate the diagram on multi-axes plots. Plot divided into alkaline 
    and subalkaline fields after Irvine and Barangar, 1971. Values used for 
    plot lines were taken from source code of GCDKit (Janousek et al., 2006).
    
    Parameters:
        SiO2: List of SiO2 values (wt. %)
        Na2O: List of Na2O values (wt. %)
        K2O: List of K2O values (wt. %)
        ax: Axes on which to plot the diagram
        first: Empty list by default. If empty, lines/labels will plot
    
    Returns:
        ax: Axes with TAS plotted
    """
    if ax is None:
        ax = plt.gca()
    # Calculate total alkalis
    alkalis = Na2O + K2O
    
    #Plot data
    ax.scatter(SiO2,alkalis, **plt_kwargs)
    
    # Check if first empty to avoid repeat plotting of TAS grid/labels
    if first == []:
    # Create lines
        line1 = [[30,41,41,45,48.4,52.5,30],[0,0,7,9.4,11.5,14,24.15]]
        line2 = [[41,45,45,41],[0,0,3,3]]
        line3 = [[5,52,52,45],[0,0,5,5]]
        line4 = [[52,57,57,52],[0,0,5.9,5]]
        line5 = [[57,63,63,57],[0,0,7,5.9]]
        line6 = [[63,77,69,63],[0,0,8,7]]
        line7 = [[77,100,100,69,69],[0,0,25,25,8]]
        line8 = [[45,52,49.4],[5,5,7.3]]
        line9 = [[52,57,53,49.4],[5,5.9,9.3,7.3]]
        line10 = [[57,63,57.6,53],[5.9,7,11.7,9.3]]
        line11 = [[63,69,69,57.6],[7,8,17.73,11.7]]
        line12 = [[41,45,45,49.4,45,41],[3,3,5,7.3,9.4,7]]
        line13 = [[49.4,53,48.4,45],[7.3,9.3,11.5,9.4]]
        line14 = [[53,57.6,52.5,48.4],[9.3,11.7,14,11.5]]
        line15 = [[57.6,69,30],[11.7,17.73,24.15]]
        lines = [line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,
                 line11,line12,line13,line14,line15]
                      
        # Create abbreviated labels
        labelsx = [43,48.5,54.8,59.9,67,75,63.5,57.8,52.95,49.2,45,49.2,53,57,
                   43]
        labelsy = [1.55,2.8,3,3,3,8,11,8.5,7,5.65,7,9.3,11.5,14,12]
        labeltext = ['PB','B','BA','A','D','R','T/TD','TA','BTA','TB',
                     'TEP/\nBSN','PHT','TPH','PH','FOI']
                     
        # Create subalkaline/alkaline fields
        subalkx = [39.2,40,43.2,45,48,50,53.7,55,60,65,77.4] 
        subalky = [0,0.4,2,2.8,4,4.75,6,6.4,8,8.8,10]
        
        
        #Plot Subalkaline/Alkaline line without text
        ax.plot(subalkx,subalky,'r--')

        #Set axes limits
        ax.set_xlim(35,80)
        ax.set_ylim(0,16)
        
        for z in range(15): #loop through TAS lines
            ax.plot(lines[z][0],lines[z][1],'k')
            ax.text(labelsx[z],labelsy[z],labeltext[z],color='k',
                    ha='center',va='center',fontsize=6)
        
        # Set small fonts
        ax.set_xlabel('$\mathregular{SiO_2}$ (wt. %)',fontsize=8)
        ax.set_ylabel('$\mathregular{Na{_2}O + K{_2}O}$ (wt. %)',fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        
        # Avoid repeat grid plotting
        first.append('Not First')      
    return(ax)

def cabanis(Tb,Th,Ta,ax=None,grid=False,first=[],**plt_kwargs):
    """
    Plot Th-3Tb-2Ta diagram of Cabanis and Thieblemont (1988).
    
    Parameters:
        Tb: List of Tb values
        Th: List of Th values
        Ta: List of Ta values
        ax: Axes on which to plot, requires "ternary" projection from mpltern
        grid: Boolean for whether to add grid to diagram
        first: Empty list by default. If empty and grid is True, plot grid
    
    Returns:
        ax: Axes with diagram plotted
    """
    if ax is None:
        ax = plt.gca()
    
    # Calculate 3Tb and 2Ta
    Tb3 = Tb*3
    Ta2 = Ta*2
    
    # Set plot labels
    ax.set_tlabel('3Tb',fontsize=8)
    ax.set_llabel('Th',fontsize=8)
    ax.set_rlabel('2Ta',fontsize=8)
    
    # Plot grid
    if (grid==True) & (first==[]):
        t, l, r = get_triangular_grid()
        ax.triplot(t, l, r,color='gray',linestyle='--')
        first.append('NotFirst')
    
    # Plot data
    ax.scatter(Tb3,Th,Ta2,**plt_kwargs)
    
    # Set plot labels
    ax.set_tlabel('3Tb',fontsize=8)
    ax.set_llabel('Th',fontsize=8)
    ax.set_rlabel('2Ta',fontsize=8)
    
    # Remove plot ticks
    ax.taxis.set_ticks([])
    ax.laxis.set_ticks([])
    ax.raxis.set_ticks([])
    
    return(ax)

def cabanisd(Tb,Th,Ta,ax=None,grid=False,**plt_kwargs):
    """
    Plot Th-3Tb-2Ta diagram of Cabanis and Thieblemont (1988) as KDE.
    
    Uses KDE functionality of pyrolite (Williams et al., 2020).
    
    Parameters:
        Tb: List of Tb values
        Th: List of Th values
        Ta: List of Ta values
        ax: Axes on which to plot, requires "ternary" projection from mpltern
        grid: Boolean for whether to add grid to diagram
    
    Returns:
        ax: Axes with diagram plotted
    """
    if ax is None:
        ax = plt.gca()
    #calculate values
    Tb3 = Tb*3
    Ta2 = Ta*2
    Th1 = Th*1
    
    #Plot Grid
    if grid==True:
        t, l, r = get_triangular_grid()
        ax.triplot(t, l, r,color='gray',linestyle='--')
    
    #Make dataframe and plot
    df = pd.concat([Tb3,Th1,Ta2],axis=1)
    df.pyroplot.density(ax=ax,**plt_kwargs)
    
    #Set plot labels
    ax.set_tlabel('3Tb',fontsize=8)
    ax.set_llabel('Th',fontsize=8)
    ax.set_rlabel('2Ta',fontsize=8)
    
    ax.taxis.set_ticks([])
    ax.laxis.set_ticks([])
    ax.raxis.set_ticks([])

    return(ax)

def harker(df,fig=None,axs=None,**plt_kwargs):
    #Need to input pandas dataframe with major oxides to plot harker diagrams. Also
    #need all Fe to be in FeOt
    if fig is None:
        fig, axs = plt.subplots(4,2, sharex=True, figsize=(6.5,9),dpi=300)
    plt.setp(axs,xlim=(40,75))
    oxides = ['TiO2','Al2O3','FeOt','P2O5','CaO','MgO','Na2O','K2O']
    ylims = [(0,2.5),(12,22),(0,15),(0,1.2),(0,15),(0,10),(0,7),(0,4)]   
    for x in range(4): #For loops for oxides
        for y in range(2):
            df.plot.scatter(x = 'SiO2',y = oxides[2*x+y], ax=axs[x][y],ylim = ylims[2*x+y],
                            **plt_kwargs)
            #axs[x][y].set_xlabel(fontsize=8)
            #axs[x][y].set_ylabel(fontsize=8)
            axs[x][y].tick_params(axis='both', which='major', labelsize=6)
    plt.tight_layout()
    return(fig,axs)

def spiders(df,**plt_kwargs):
    #Need to input pandas dataframe
    PTioxides = df[['P2O5','TiO2']] #isolate oxides only
    pti = PTioxides.pyrochem.convert_chemistry(to=["P", "Ti"]) #Convert P2O and TiO2
    pti_ppm = pti.pyrochem.scale('wt%','ppm')
    if pd.Series(['P', 'Ti']).isin(df.columns).all():    
        df.update(pti_ppm)   
        print(1)
    elif 'P' in df.columns:
        df.update(pti_ppm)
        df = pd.concat([df,pti_ppm['Ti']],axis=1)
        print(2)
    elif 'Ti' in df.columns:
        df.update(pti_ppm)
        df = pd.concat([df,pti_ppm['P']],axis=1)
        print(3)
    else:
        df = pd.concat([df,pti_ppm],axis=1)
        print(4)
    trace = df[["La", "Ce", "Pr", "Nd", "Sm", "Eu",
                  "Gd", "Tb", "Dy", "Ho", "Er", "Yb","Lu","Th","Nb",
                  "Ta","P","Zr","Hf","Ti","Y"]] #NOTE: Removed Pm and Tm

    norm = trace.pyrochem.normalize_to(reference="PM_SM89", units="ppm")
    
    sm89 = ["La", "Ce", "Pr", "Nd", "Sm", "Eu",
        "Gd", "Tb", "Dy", "Ho", "Er", "Yb", "Lu"]

    sm95 = ["Th", "Nb", "Ta", "La", "Ce", "P", "Nd",
        "Zr", "Hf", "Sm", "Eu", "Ti", "Gd", "Tb", "Y", "Yb"]
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    plt.setp(ax, ylim=(0.1,1000))
    
    norm.pyroplot.spider(
        ax=ax[0],
        **plt_kwargs,
        unity_line=True, 
        components=sm89,   
    )
    
    norm.pyroplot.spider(
        ax=ax[1],
        **plt_kwargs,
        unity_line=True,
        components=sm95,  
    )
    
    ax[0].set_title('REE Elements')
    ax[1].set_title('Incompatible Elements')
    
    return(fig)

def REE(df,ax=None,**plt_kwargs):
    if ax is None:
        ax = plt.gca()
    PTioxides = df[['P2O5','TiO2']] #isolate oxides only
    pti = PTioxides.pyrochem.convert_chemistry(to=["P", "Ti"]) #Convert P2O and TiO2
    pti_ppm = pti.pyrochem.scale('wt%','ppm')
    if pd.Series(['P', 'Ti']).isin(df.columns).all():    
        df.update(pti_ppm)   
        print(1)
    elif 'P' in df.columns:
        df.update(pti_ppm)
        df = pd.concat([df,pti_ppm['Ti']],axis=1)
        print(2)
    elif 'Ti' in df.columns:
        df.update(pti_ppm)
        df = pd.concat([df,pti_ppm['P']],axis=1)
        print(3)
    else:
        df = pd.concat([df,pti_ppm],axis=1)
        print(4)
    
    sm89 = ["La", "Ce", "Pr", "Nd", "Sm", "Eu",
        "Gd", "Tb", "Dy", "Ho", "Er", "Yb", "Lu"]
    
    trace = df[sm89]
    
    norm = trace.pyrochem.normalize_to(reference="PM_SM89", units="ppm")
    
    norm.pyroplot.spider(
    ax=ax,
    **plt_kwargs,
    unity_line=True, 
    components=sm89,   
    )
    
    ax.set_ylim(0.1,1000)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_ylabel('Sample/Primitive Mantle',fontsize=8)
    
    return(ax)

def immobile(df,ax=None,**plt_kwargs):
    if ax is None:
        ax = plt.gca()
    PTioxides = df[['P2O5','TiO2']] #isolate oxides only
    pti = PTioxides.pyrochem.convert_chemistry(to=["P", "Ti"]) #Convert P2O and TiO2
    pti_ppm = pti.pyrochem.scale('wt%','ppm')
    if pd.Series(['P', 'Ti']).isin(df.columns).all():    
        df.update(pti_ppm)   
        print(1)
    elif 'P' in df.columns:
        df.update(pti_ppm)
        df = pd.concat([df,pti_ppm['Ti']],axis=1)
        print(2)
    elif 'Ti' in df.columns:
        df.update(pti_ppm)
        df = pd.concat([df,pti_ppm['P']],axis=1)
        print(3)
    else:
        df = pd.concat([df,pti_ppm],axis=1)
        print(4)
    
    sm95 = ["Th", "Nb", "Ta", "La", "Ce", "P", "Nd",
        "Zr", "Hf", "Sm", "Eu", "Ti", "Gd", "Tb", "Y", "Yb"]
    
    trace = df[sm95]
    
    norm = trace.pyrochem.normalize_to(reference="PM_SM89", units="ppm")
    
    norm.pyroplot.spider(
    ax=ax,
    **plt_kwargs,
    unity_line=True, 
    components=sm95,   
    )
    
    ax.set_ylim(0.1,1000)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_ylabel('Sample/Primitive Mantle',fontsize=8)
    
    return(ax)
    
    

def NdSr(eNd,Sr,init=False,ax=None,**plt_kwargs):
    if ax is None:
        ax = plt.gca()
    ax.scatter(Sr,eNd,**plt_kwargs)
    if init==False:
        ax.set_xlabel('$\mathregular{^{87}Sr/^{86}Sr}$',fontsize=8)
        ax.set_ylabel('\u03B5Nd',fontsize=8)
    elif init==True:
        ax.set_xlabel('$\mathregular{^{87}Sr/^{86}Sr_i}$',fontsize=8)
        ax.set_ylabel('$\mathregular{\u03B5Nd_i}$',fontsize=8)
    ax.set_xlim(0.700,0.712)
    ax.set_ylim(-12,15)
    ax.axvline(0.7045,c='gray',zorder=0)
    ax.axhline(0,c='gray',zorder=0)
    ax.tick_params(axis='both', which='major', labelsize=6)
    return(ax)

def NdSrd(df,init=False,ax=None,**plt_kwargs):
    #Needs df with epsNd and Sr labels already present
    if ax is None:
        ax = plt.gca()

    if init==False:
        df.loc[:,['87Sr/86Sr','\u03B5Nd']].pyroplot.density(ax=ax,
                                                            extent=[0.700,0.712,-12,15],
                                                        vmin=0.05,bins=100,
                                                        **plt_kwargs)
        ax.set_xlabel('$\mathregular{^{87}Sr/^{86}Sr}$',fontsize=8)
        ax.set_ylabel('\u03B5Nd',fontsize=8)
       
    elif init==True:
        df.loc[:,['87Sr/86Sri','\u03B5Ndi']].pyroplot.density(ax=ax,
                                                            extent=[0.700,0.712,-12,15],
                                                        vmin=0.05,bins=100,
                                                        **plt_kwargs)
        ax.set_xlabel('$\mathregular{^{87}Sr/^{86}Sr_i}$',fontsize=8)
        ax.set_ylabel('$\mathregular{\u03B5Nd_i}$',fontsize=8)
        
    ax.set_xlim(0.700,0.712)
    ax.set_ylim(-12,15)
    ax.axvline(0.7045,c='gray',zorder=0)
    ax.axhline(0,c='gray',zorder=0)
    ax.tick_params(axis='both', which='major', labelsize=6)
    return(ax)

def subfig(fig,xloc=0,yloc=1,fontsize=16,**plt_kwargs):
    #Add subfig labels to axes in figure using axes coordinates
    axes = fig.get_axes() #get all axes in fig
    letters = list(string.ascii_lowercase)
    for x in range(len(axes)):
        axes[x].text(xloc, yloc,'('+letters[x]+')',transform=axes[x].transAxes,
                     fontsize=fontsize,va='top',**plt_kwargs)
        