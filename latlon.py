"""
Module for basic transformations involving latitude/longitude
"""
import cartopy.crs as ccrs

def dms2deg(latdeg,londeg,latmin,lonmin,latsec=0,lonsec=0):
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
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
    """
    lat = (latdeg + (latmin+latsec/60)/60)
    lon = (londeg + (lonmin+lonsec/60)/60)
    return(lat,lon)

def UTM2latlon(easting,northing,zone,south=False):
    """
    Calculate latitude/longitude from UTM.
    """
    # Set ouptput to lat/lon
    crs = ccrs.Geodetic()
    
    # Convert
    lon,lat = crs.transform_point(
        easting,northing,src_crs=ccrs.UTM(zone=zone,
                                          southern_hemisphere=south))
    
    return(lat,lon)