"""
Module for making geochemical plots.
"""
import string

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pyrolite.plot

from mpltern.ternary.datasets import get_triangular_grid
from matplotlib.patches import Polygon


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

def afm(Na2O,K2O,FeOt,MgO,ax=None,first=[],fontsize=8,
        scatter=True,density=False,scatter_kwargs={},
        density_kwargs={}):
    """
    Plot AFM diagram after Irvine and Baragar (1971)
    """
    if first==[]:
        A,F,M = afm_line()
        ax.plot(F,A,M,linestyle='--',color='black')

        # Add tholeiite and calc-alkaline
        ax.text(0.7,0.15,0.15,'Tholeiitic',ha='center', va='center',fontsize=8)
        ax.text(0.25,0.375,0.375,'Calc-Alkaline',ha='center', va='center',fontsize=8)

        first.append('NotFirst')
    
    if density:
        df = pd.DataFrame(data=[FeOt,Na2O+K2O,MgO]).T
        df.pyroplot.density(ax=ax,**density_kwargs)
    
    if scatter:
        ax.scatter(FeOt,Na2O+K2O,MgO,**scatter_kwargs)
    
    # Set labels
    ax.set_tlabel('F',fontsize=fontsize)
    ax.set_llabel('A',fontsize=fontsize)
    ax.set_rlabel('M',fontsize=fontsize)

    # Remove ticks
    ax.taxis.set_ticks([])
    ax.laxis.set_ticks([])
    ax.raxis.set_ticks([])



    return(ax)

def afm_line():
    M = np.arange(5,45,2)
    F = (
        1.5559e-12 * M**8 - 7.7142e-10 * M**7 + 1.5664e-7 * M**6
          - 1.6738e-5 * M**5 + 1.0017e-3 * M**4 - 3.2552e-2 * M**3
            + 4.7776e-1 * M**2 - 1.1085 * M + 30.0
    )
    A = 100 - M - F
    return(A,F,M)

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
    
    # Calculate 3Tb and 2Ta
    Tb3 = Tb*3
    Ta2 = Ta*2
    Th1 = Th*1
    
    # Plot grid
    if grid==True:
        t, l, r = get_triangular_grid()
        ax.triplot(t, l, r,color='gray',linestyle='--')
    
    # Make into Pandas dataframe and plot using pyrolite
    df = pd.concat([Tb3,Th1,Ta2],axis=1)
    df.pyroplot.density(ax=ax,**plt_kwargs)
    
    # Set plot labels
    ax.set_tlabel('3Tb',fontsize=8)
    ax.set_llabel('Th',fontsize=8)
    ax.set_rlabel('2Ta',fontsize=8)
    
    # Remove plot ticks
    ax.taxis.set_ticks([])
    ax.laxis.set_ticks([])
    ax.raxis.set_ticks([])

    return(ax)

def mantle_array(Th,Nb,Yb,ax=None,first=[],scatter=True,density=False,
                 scatter_kwargs={},density_kwargs={}):
    """
    Mantle array plot after Pearce, 2008
    """
    if ax is None:
        ax = plt.gca()

    if first==[]:
    # MORB-OIB Array
        x = [0.1,0.3,1000,1000,800,0.1]
        y = [0.01,0.01,48,100,100,0.01]
        xy_array = np.column_stack((x,y))

        # Arc Array
        b = (np.log10(10)-np.log10(1.2))/(np.log10(0.8)-np.log10(0.1))
        a = np.log10(10)-np.log10(0.8)
        xvals = np.arange(0,1000)
        yvals = a*np.power(xvals,b)

        pgon = Polygon(xy_array,alpha=0.2,zorder=0,color='gray')
        ax.add_patch(pgon)
        ax.plot(xvals,yvals,color='gray')

        first.append('NotFirst')

    if density:
        df = pd.DataFrame(data=[Nb/Yb,Th/Yb]).T
        df.pyroplot.density(ax=ax,logx=True,logy=True,**density_kwargs)
        
    if scatter:
        ax.scatter(Nb/Yb,Th/Yb,**scatter_kwargs)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(0.1,100)
    ax.set_ylim(0.01,10)
    ax.set_xlabel('Nb/Yb')
    ax.set_ylabel('Th/Yb')

    return(ax)

def harker(df,fig=None,axs=None,**plt_kwargs):
    """
    Plot silica variation ("Harker") diagrams for major oxides.
    
    Plots SiO2 (wt. %) against TiO2, Al2O3, FeOt, P2O5, CaO, MgO, Na2O, and
    K2O
    
    Parameters:
        df: Pandas dataframe with major oxide information. Requires Fe input
            as FeOt
        fig: Figure on which to plot diagrams
        axs: Set of axs within figure on which to plot
        
    Returns:
        fig: Figure with diagrams plotted
        axs: Axes within figure with diagrams plotted
    """
    # Create figure if not specified
    if fig is None:
        fig, axs = plt.subplots(4,2, sharex=True, figsize=(6.5,9),dpi=300)
   
   # Set axs limits and oxides 
    plt.setp(axs,xlim=(40,75)) 
    oxides = ['TiO2','Al2O3','FeOt','P2O5','CaO','MgO','Na2O','K2O']
    ylims = [(0,2.5),(12,22),(0,15),(0,1.2),(0,15),(0,10),(0,7),(0,4)]   
    
    # Nested for loops to plot each oxide
    for x in range(4): 
        for y in range(2):
            df.plot.scatter(x = 'SiO2',y = oxides[2*x+y], ax=axs[x][y],ylim = ylims[2*x+y],
                            **plt_kwargs)
            axs[x][y].tick_params(axis='both', which='major', labelsize=6)
    plt.tight_layout()
    return(fig,axs)

def spiders(df,**plt_kwargs):
    """
    Plots figure of rare earth element plot and immobile element plot.
    
    Plots rare earth element plot and imbbolie element plot after Pearce,
    2014. Rare earth element plot does not include Pm or Tm, given low
    availability in used data. Somewhat deprecated in favor of REE and
    immobile below, which are more flexible for multi-axes figures.
    
    Parameters:
        df: Pandas dataframe with necessary trace element data
    
    Retruns:
        fig: Figure with both plots
    """ 
    # Convert P and Ti from oxides to ppm, if needed, using pyrolite
    PTioxides = df[['P2O5','TiO2']] #isolate oxides only
    pti = PTioxides.pyrochem.convert_chemistry(to=["P", "Ti"]) #Convert
    pti_ppm = pti.pyrochem.scale('wt%','ppm')
    
    # Set of if statements for how to proceed depending on if P/Ti were
    # previously reported
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
    
    # List of elements for both plots
    trace = df[["La", "Ce", "Pr", "Nd", "Sm", "Eu","Gd", "Tb", "Dy", "Ho",
                   "Er", "Yb","Lu","Th","Nb","Ta","P","Zr","Hf","Ti","Y"]]
                  
    # Normalize to primitive mantle
    norm = trace.pyrochem.normalize_to(reference="PM_SM89", units="ppm")
    
    # Define rare earth elements
    ree = ["La", "Ce", "Pr", "Nd", "Sm", "Eu",
        "Gd", "Tb", "Dy", "Ho", "Er", "Yb", "Lu"]

    # Define immobile elements
    imm = ["Th", "Nb", "Ta", "La", "Ce", "P", "Nd",
        "Zr", "Hf", "Sm", "Eu", "Ti", "Gd", "Tb", "Y", "Yb"]
    
    # Set up figure and plot
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    plt.setp(ax, ylim=(0.1,1000))
    
    norm.pyroplot.spider(
        ax=ax[0],
        **plt_kwargs,
        unity_line=True, 
        components=ree,   
    )
    
    norm.pyroplot.spider(
        ax=ax[1],
        **plt_kwargs,
        unity_line=True,
        components=imm,  
    )
    
    # Add titles
    ax[0].set_title('REE Elements')
    ax[1].set_title('Incompatible Elements')
    
    return(fig)

def REE(df,ax=None,**plt_kwargs):
    """
    Plot rare earth element digaram, normalized to primitive mantle.
    
    Plot normalized to primitive mantle values of Sun and McDonough, 1989.
    Does not contain seldom-used Pm or Tm. Uses pyrolite extensively.
    
    Parameters:
        df: Pandas dataframe with geochemical data.
        ax: Axes on which to plot diagram
    
    Returns:
        ax: Axes with diagram plotted
    """
    if ax is None:
        ax = plt.gca()
   
    # Convert P and Ti from oxides to ppm, if needed, using pyrolite
    PTioxides = df[['P2O5','TiO2']] #isolate oxides only
    pti = PTioxides.pyrochem.convert_chemistry(to=["P", "Ti"]) #Convert
    pti_ppm = pti.pyrochem.scale('wt%','ppm')
    
    # Set of if statements for how to proceed depending on if P/Ti were
    # previously reported
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
    
    # Set rare earth elements
    ree = ["La", "Ce", "Pr", "Nd", "Sm", "Eu",
        "Gd", "Tb", "Dy", "Ho", "Er", "Yb", "Lu"]
    
    # Get values from dataframe and normalize
    trace = df[ree] 
    norm = trace.pyrochem.normalize_to(reference="PM_SM89", units="ppm")
    
    norm.pyroplot.spider(
    ax=ax,
    **plt_kwargs,
    unity_line=True, 
    components=ree,   
    )
    
    ax.set_ylim(0.1,1000)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_ylabel('Sample/Primitive Mantle',fontsize=8)
    
    return(ax)

def immobile(df,ax=None,**plt_kwargs):
    """
    Plot immobile element digaram, normalized to primitive mantle.
    
    Plot normalized to primitive mantle values of Sun and McDonough, 1989.
    After Pearce, 2014. Uses pyrolite extensively.
    
    Parameters:
        df: Pandas dataframe with geochemical data.
        ax: Axes on which to plot diagram
    
    Returns:
        ax: Axes with diagram plotted
    """
    if ax is None:
        ax = plt.gca()
        
    # Convert P and Ti from oxides to ppm, if needed, using pyrolite
    PTioxides = df[['P2O5','TiO2']] #isolate oxides only
    pti = PTioxides.pyrochem.convert_chemistry(to=["P", "Ti"]) #Convert
    pti_ppm = pti.pyrochem.scale('wt%','ppm')
    
    # Set of if statements for how to proceed depending on if P/Ti were
    # previously reported
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
    
    # Set immobile elements
    imm = ["Th", "Nb", "Ta", "La", "Ce", "P", "Nd",
        "Zr", "Hf", "Sm", "Eu", "Ti", "Gd", "Tb", "Y", "Yb"]
    
    # Get values from dataframe and normalize
    trace = df[imm]
    norm = trace.pyrochem.normalize_to(reference="PM_SM89", units="ppm")
    
    norm.pyroplot.spider(
    ax=ax,
    **plt_kwargs,
    unity_line=True, 
    components=imm,   
    )
    
    ax.set_ylim(0.1,1000)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_ylabel('Sample/Primitive Mantle',fontsize=8)
    
    return(ax)
    
def NdSr(eNd,Sr,init=False,ax=None,**plt_kwargs):
    """
    Plot diagram of epsilon Nd vs. 87Sr/86Sr.
    
    Parameters:
        eNd: Values for epsilon Nd
        Sr: Values for 87Sr/86Sr
        init: Boolean for if values are initial values
        ax: Axes on which to plot diagram
        
    Returns:
        ax: Axes with diagram plotted
    """
    if ax is None:
        ax = plt.gca()
    
    ax.scatter(Sr,eNd,**plt_kwargs)
    
    # Set labels according to whether initial or present day values
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
    """
    Plot diagram of epsilon Nd vs. 87Sr/86Sr as KDE using pyrolite.
    
    Parameters:
        df: Pandas dataframe with epsilon Nd and 87Sr/86Sr. Requires columns
            labeled '87Sr/86Sr' and '\u03B5Nd'.
        init: Boolean for if values are initial values
        ax: Axes on which to plot diagram
        
    Returns:
        ax: Axes with diagram plotted
    """
    if ax is None:
        ax = plt.gca()

    # Plot using appropriate labels, depending on whether values are initial.
    if init==False:
        df.loc[:,['87Sr/86Sr','\u03B5Nd']].pyroplot.density(
        ax=ax,
        extent=[0.700,0.712,-12,15],                                                
        vmin=0.05,
        bins=100,                                            
        **plt_kwargs
        )
        ax.set_xlabel('$\mathregular{^{87}Sr/^{86}Sr}$',fontsize=8)
        ax.set_ylabel('\u03B5Nd',fontsize=8)
       
    elif init==True:
        df.loc[:,['87Sr/86Sri','\u03B5Ndi']].pyroplot.density(
        ax=ax,
        extent=[0.700,0.712,-12,15],                                                
        vmin=0.05,
        bins=100,                                            
        **plt_kwargs
        )    
        ax.set_xlabel('$\mathregular{^{87}Sr/^{86}Sr_i}$',fontsize=8)
        ax.set_ylabel('$\mathregular{\u03B5Nd_i}$',fontsize=8)
        
    ax.set_xlim(0.700,0.712)
    ax.set_ylim(-12,15)
    
    ax.axvline(0.7045,c='gray',zorder=0)
    ax.axhline(0,c='gray',zorder=0)
    
    ax.tick_params(axis='both', which='major', labelsize=6)
    
    return(ax)

def subfig(fig,xloc=0,yloc=1,fontsize=16,**plt_kwargs):
    """
    Add subfigure labels to axes in figure using axes coordinates
    
    Parameters:
        fig: Figure on which to apply labels
        xloc: X location for label, in axes coordinates (0-1)
        yloc: Y location for label, in axes coordinates (0-1)
        fontsize: Font size for axes labels
        
    Returns:
        fig: Figure with labels applied
    """
    axes = fig.get_axes() # Get all axes in figure
    letters = list(string.ascii_lowercase)
    for x in range(len(axes)):
        axes[x].text(xloc, yloc,'('+letters[x]+')',transform=axes[x].transAxes,
                     fontsize=fontsize,va='top',**plt_kwargs)
    return(fig)    