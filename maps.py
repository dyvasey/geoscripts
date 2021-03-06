"""
Module of mapping functions for use with Cartopy
"""
import string
import json

import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.geodesic as cgeo
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from shapely import geometry
from shapely.geometry.polygon import Polygon
import pandas as pd

def bath(res='10m',ax=None):
    """
    Plot bathymetry from Natural Earth as set of blue vector polygons.
    
    Parameters:
        res: Resolution of Natural Earth data
        ax: Axes on which to plot
    
    Returns:
        ax: Axes with bathymetry plotted
    """
    if ax is None:
        ax = plt.gca()
    
    depths = np.append(np.arange(10000,999,-1000),[200,0]) # List of depths
    letters = string.ascii_uppercase # Letters for filenames
    cmap = matplotlib.cm.get_cmap('Blues') # Set colormap
    colors = cmap(np.linspace(1,0.5,12)) # Define colors from colormap
    
    # Iterate through depths starting at 0 m and plot
    for x in reversed(range(len(depths))):
        step = 'bathymetry_'+ letters[x] + '_' + str(depths[x]) # File name
        ax.add_feature(cf.NaturalEarthFeature('physical',step,res),
                       color=colors[x],zorder=1)
    return(ax)

def boundingbox(box,crs,ax=None,**kwargs):
    """
    Plot bounding box for simple boxes where north is up.
    
    Parameters:
        box: list of min/max values for box - xmin, xmax, ymin, ymax
        crs: Cartopy projection for box values
        ax: Axes on which to plot bounding box
    
    Returns:
        ax: Axes with bounding box plotted
    """
    if ax is None:
        ax = plt.gca()
    geo = geometry.box(box[0],box[2],box[1],box[3])
    ax.add_geometries([geo], crs=crs,**kwargs)
    return(ax)

def bb_irreg(lon,lat,ax=None,**kwargs):
    """
    Plot irregular bounding box with vertices defined as longitude/latitude.
    
    Parameters:
        lon: Longitude values in decimal degrees
        lat: latitude values in decimal degrees
        ax: Axes on which to plot bounding box
    
    Returns:
        ax: Axes with bounding box plotted
    """
    if ax is None:
        ax = plt.gca()
    
    # Create Shapely polygon of values
    pgon = Polygon(((lon[0], lat[0]),
        (lon[1], lat[1]),
        (lon[2], lat[2]),
        (lon[3], lat[3]),
        (lon[0], lat[0])))
    ax.add_geometries([pgon], crs=ccrs.PlateCarree(),**kwargs)
    return(ax)

def bb_auto(source_ax,ax=None,**kwargs):
    """
    Plot bounding box using boundaries of existing axes.
    
    Parameters:
        source_ax: Axes object from which boundaries will be drawn
        ax: Axes on which to plot bounding box
    """
    if ax is None:
        ax = plt.gca()
    
    # Convert source axes corners to values in the display coordinate system
    bleft = source_ax.transAxes.transform((0,0))
    bright = source_ax.transAxes.transform((1,0))
    tleft = source_ax.transAxes.transform((0,1))
    tright = source_ax.transAxes.transform((1,1))
    
    # Convert display coordinate system values to data (map) coordinate system
    bl_ll = source_ax.transData.inverted().transform(bleft)
    br_ll = source_ax.transData.inverted().transform(bright)
    tl_ll = source_ax.transData.inverted().transform(tleft)
    tr_ll = source_ax.transData.inverted().transform(tright)
    
    # Make coordinates into Shapely polygon
    pgon = Polygon((bl_ll,
        br_ll,
        tr_ll,
        tl_ll,
        bl_ll))
    
    # Add bounding box to Axes, indicating that data are in source axes coords
    ax.add_geometries([pgon], crs=source_ax.projection,**kwargs)
    
    return(ax)
    
def smlabels(ax,box,step=1):
    """
    Add small latitude/longitude labels to map.
    
    Parameters:
        ax: Axes on which to add labels
        box: Values across which to make labels - xmin, xmax, ymin, ymax
        step: How wide steps should be between labels, in data coordinates.
    
    Returns:
        gl: Gridlines ojbect to add to axes
    """
    gl = ax.gridlines(draw_labels=True)
    gl.bottom_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    
    # Add x labels
    gl.xlocator = mticker.FixedLocator(
        np.arange(round(box[0]-step/2,None),round(box[1]+step/2,None),step)                                     
        )
    
    # Add y labels
    gl.ylocator = mticker.FixedLocator(
        np.arange(round(box[2]-step/2,None),round(box[3]+step/2,None),step)
        )
    
    return(gl)

def shpplt_simple(shp,ax=None,crs=ccrs.PlateCarree(),**kwargs):
    """
    Plot .shp file from path using uniform colors/style
    
    Parameters:
        shp: Path to .shp file
        ax: Axes on which to plot polygon
        crs: Cartopy projection for .shp file
    
    Retruns:
        ax: Axes with .shp file plotted
    """
    if ax is None:
        ax = plt.gca()
    # Convert .shp file to Shapely Feature
    feature = ShapelyFeature(Reader(shp).geometries(),crs,**kwargs)
    
    ax.add_feature(feature)
    
    return(ax)

def shpplt(shp,colors,ax=None,crs=ccrs.PlateCarree(),field='type',
           numbered=True,**kwargs):
    """
    Plot polygon .shp file from path using variable colors.
    
    Requires that .shp file have field with numerical values that
    will correspond to color.
    
    Parameters:
        shp: Path to .shp file
        colors: List or dict of colors corresponding to field in .shp file
        ax: Axes on which to plot polygon
        crs: Cartopy projection for .shp file
        field: Name of field with values used for color
    
    Retruns:
        df: Pandas dataframe of attribute table from .shp file
    """
    reader = Reader(shp) # Creates reader object
    units = reader.records() # Gets records from reader object
    lst = [] # Set up empty list
    
    for unit in units: # Iterate over each object in .shp file
        # Get type number from attributes to assign color, requires 'type'
        # field in imported .shp file
        t = unit.attributes[field] 
        geo = unit.geometry # Get geometry to plot
        
        # Plot geometry using field attribute to assign color
        
        # If field sequential integers, adapt for Python counting
        if numbered is True:
            ax.add_geometries([geo],crs,facecolor=colors[t-1],**kwargs)
        # If field non-sequential, use color dictionary
        elif numbered is False:
            ax.add_geometries([geo],crs,facecolor=colors[t],**kwargs)
        
        # Add attributes to list to create dataframe
        lst.append(unit.attributes) 
    
    df = pd.DataFrame.from_records(lst)  #create dataframe of attribute table
    
    return(df)

def scalebar(length,slon,slat,az=90,label=True,ax=None,**kwargs):
    """
    Plot scalebar of given length in meters.
    
    Parameters:
        length: Length of scalebar in meters
        slon: Starting longitude (decimal degrees) for scalebar
        slat: Starting latitude (decimal degrees) for scalebar
        az = Azimuth of scalebar
        label: Boolean for whether to label length of scalebar in km
        ax: Axes on which to plot scalebar
    
    Return:
        ax: Axes with scalebar plotted
    """
    if ax is None:
        ax = plt.gca()
    
    geodesic = cgeo.Geodesic() # Set up geodesic calculations
    
    # Calculate endpoint for given distance
    end = geodesic.direct(
        points=[slon,slat],azimuths=az,distances=length).base[0]
    elon = end[0]
    elat = end[1]
    clon = (slon+elon)/2
    clat = (slat+elat)/2
    
    # Plot line from start to end
    ax.plot([slon,elon],[slat,elat],transform=ccrs.Geodetic(),
            **kwargs,linewidth=3)
    
    # Add label with number of km
    if label==True:    
        # Get map projection from axes
        crs = ax.projection
        # Transform lat-lon into axes coordinates
        tlon,tlat = crs.transform_point(clon,clat,src_crs=ccrs.Geodetic())
        
        # Add label as annotation
        ax.annotate(text=str(round(length/1000,None))+' km',xy=(tlon,tlat),
                xytext=(0,3),xycoords='data',textcoords='offset points',
                fontsize=7,ha='center')
    
    return(ax)

def narrow(lon,lat,ax=None,lfactor=1,**kwargs):
    """
    Plot north arrow.
    
    Parameters:
        lon: Starting longitude (decimal degrees) for arrow
        lat: Starting latitude (ecimal degrees) for arrow
        ax: Axes on which to plot arrow
        lfactor: Length factor to increase/decrease arrow length
    
    Returns:
        ax: Axes with arrow plotted
    """
    if ax is None:
        ax = plt.gca()
    
    geodesic = cgeo.Geodesic() # Set up geodesic calculations
    
    # Get map projection from axes
    crs = ax.projection
    
    # Get geodetic projection for lat/lon - do not confuse with geodesic
    gdet = ccrs.Geodetic()
    
    # Get axes extent and convert to lat/lon
    x1,x2,y1,y2 = ax.get_extent()
    tlx,tly = gdet.transform_point(x1,y2,src_crs=crs) 
    blx,bly = gdet.transform_point(x1,y1,src_crs=crs) 
    diff = abs(bly-tly) # Get x coverage of plot in decimal degrees
    
    # Get arrow endpoint scaled by diff and lfactor
    end = geodesic.direct(
        points=[lon,lat],azimuths=0,distances=lfactor*diff*2*10**4).base[0]

    # Transform lat-lon into axes coordinates
    xstart,ystart = crs.transform_point(lon,lat,src_crs=ccrs.Geodetic())
    
    # Get X-Y coordinates of endpoint
    xend,yend = crs.transform_point(end[0],end[1],src_crs=ccrs.Geodetic())
    
    # Plot arrow as annotation
    ax.annotate("",xy=(xstart,ystart),xycoords='data',xytext=(xend,yend),
                textcoords='data',arrowprops=dict(arrowstyle="<|-",
                                                  connectionstyle="arc3"))
    # Add N to arrow
    ax.text(xend,yend,'N',fontsize=7,ha='center')
    
    return(ax)

def lyr_color(js):
    """
    Import JSON of .lyr file and extract dictionary of value/color.
    
    Requires using arcpy in ArcMap console to first extract .lyr file as JSON. 
    See arc_scripts.py.
    """
    # Open the JSON file
    with open(js,"r") as file:
        text = json.load(file)

    # Convert str to dictionary
    main = json.loads(text)
    renderer = main['renderer']
    unique = renderer['uniqueValueInfos']
    
    d = {} # Create empty dictionary
    
    # Extract value and color
    for unit in unique:
        value = unit['value']
        value_int = int(value)
        color = unit['symbol']['color']
        color_norm = [x/255 for x in color ]
        color_tuple = tuple(color_norm)
        d[value_int]=color_tuple
    
    return(d)
    