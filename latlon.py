"""
Module for basic transformations involving latitude/longitude
"""
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