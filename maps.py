"""
Module of mapping functions for use with cartopy
"""
import string

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
    #bounding box for simple boxes where north is up
    if ax is None:
        ax = plt.gca()
    geo = geometry.box(box[0],box[2],box[1],box[3])
    ax.add_geometries([geo], crs=crs,**kwargs)
    return(ax)

def bb_irreg(lon,lat,ax=None,**kwargs):
    #bounding box where vertices can be defined as lat lon
    if ax is None:
        ax = plt.gca()
    pgon = Polygon(((lon[0], lat[0]),
        (lon[1], lat[1]),
        (lon[2], lat[2]),
        (lon[3], lat[3]),
        (lon[0], lat[0])))
    ax.add_geometries([pgon], crs=ccrs.PlateCarree(),**kwargs)
    return(ax)

def bb_auto(source_ax,ax=None,**kwargs):
    if ax is None:
        ax = plt.gca()
    bleft = source_ax.transAxes.transform((0,0))
    bright = source_ax.transAxes.transform((1,0))
    tleft = source_ax.transAxes.transform((0,1))
    tright = source_ax.transAxes.transform((1,1))
    
    bl_ll = source_ax.transData.inverted().transform(bleft)
    br_ll = source_ax.transData.inverted().transform(bright)
    tl_ll = source_ax.transData.inverted().transform(tleft)
    tr_ll = source_ax.transData.inverted().transform(tright)
    pgon = Polygon((bl_ll,
        br_ll,
        tr_ll,
        tl_ll,
        bl_ll))
    ax.add_geometries([pgon], crs=source_ax.projection,**kwargs)
    return(ax)
    

def smlabels(ax,box,step=1):
    #function to add small labels for lat/lon
    gl = ax.gridlines(draw_labels=True)
    gl.bottom_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xlocator = mticker.FixedLocator(np.arange(round(box[0]-step/2,None),
                                             round(box[1]+step/2,None),step))
    gl.ylocator = mticker.FixedLocator(np.arange(round(box[2]-step/2,None),
                                                 round(box[3]+step/2,None),step))
    return(gl)

def shpplt_simple(shp,ax=None,crs=ccrs.PlateCarree(),**kwargs):
    #function to plot ship file from path with uniform colors.
    if ax is None:
        ax = plt.gca()
    feature = ShapelyFeature(Reader(shp).geometries(),crs,**kwargs)
    ax.add_feature(feature)
    return(ax)

def shpplt(shp,colors,ax=None,crs=ccrs.PlateCarree(),**kwargs):
    #Give shapefile path (shp), list of colors for each type.
    reader = Reader(shp) #creates reader object
    units = reader.records() #gets records from reader object
    lst = [] #set up empty list
    for unit in units: #iterate over each object in shp file
        t = unit.attributes['type'] #get type number from attributes to assign color
        geo = unit.geometry # get geometry to plot
        ax.add_geometries([geo],crs,facecolor=colors[t-1],**kwargs) #plot geometry using type attribute to assign color
        lst.append(unit.attributes) #add attributes to list to create dataframe
    df = pd.DataFrame.from_records(lst)  #create dataframe of attribute table
    return(df)

def scalebar(length,slon,slat,az=90,label=True,ax=None,**kwargs):
    #function to create scalebar of given length in meters
    if ax is None:
        ax = plt.gca()
    geodesic = cgeo.Geodesic() #set up geodesic calculations
    #calculate endpoint for given distance
    end = geodesic.direct(points=[slon,slat],azimuths=az,distances=length).base[0]
    elon = end[0]
    elat = end[1]
    clon = (slon+elon)/2
    clat = (slat+elat)/2
    #plot line from start to end
    ax.plot([slon,elon],[slat,elat],transform=ccrs.Geodetic(),**kwargs,linewidth=3)
    if label==True:
        #add label with number of km
        #get map projection from axes
        crs = ax.projection
        #transform lat-lon into axes coordinates
        tlon,tlat = crs.transform_point(clon,clat,src_crs=ccrs.Geodetic())
        ax.annotate(text=str(round(length/1000,None))+' km',xy=(tlon,tlat),
                xytext=(0,3),xycoords='data',textcoords='offset points',
                fontsize=7,ha='center')
    return(ax)

def narrow(lon,lat,ax=None,lfactor=1,**kwargs):
    #north arrow plotter
    if ax is None:
        ax = plt.gca()
    geodesic = cgeo.Geodesic() #set up geodesic calculations
    #get map projection from axes
    crs = ax.projection
    gdet = ccrs.Geodetic()
    x1,x2,y1,y2 = ax.get_extent()
    #left,right = ax.get_xlim() #get x limit of graph
    tlx,tly = gdet.transform_point(x1,y2,src_crs=crs) #convert top left to latlon
    blx,bly = gdet.transform_point(x1,y1,src_crs=crs) #convert bottom left to latlon
    diff = abs(bly-tly) #get x coverage of graph in degrees
    #get endpoint scaled by diff and lfactor
    end = geodesic.direct(points=[lon,lat],azimuths=0,distances=lfactor*diff*2*10**4).base[0]

    #transform lat-lon into axes coordinates
    xstart,ystart = crs.transform_point(lon,lat,src_crs=ccrs.Geodetic())
    #x-y coordinates of endpoint
    xend,yend = crs.transform_point(end[0],end[1],src_crs=ccrs.Geodetic())
    #plot arrow
    ax.annotate("",xy=(xstart,ystart),xycoords='data',xytext=(xend,yend),
                textcoords='data',arrowprops=dict(arrowstyle="<|-",
                                                  connectionstyle="arc3"))
    #add N
    ax.text(xend,yend,'N',fontsize=7,ha='center')
    return(ax)