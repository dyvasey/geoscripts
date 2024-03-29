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
    lat = (abs(latdeg) + (abs(latmin)+abs(latsec)/60)/60)
    lon = (abs(londeg) + (abs(lonmin)+abs(lonsec)/60)/60)
    
    if isinstance(latdeg,(int,float)):
        if latdeg<0:
            lat = -lat
        if londeg<0:
            lon = -lon
    
    else:
        for k,l in enumerate(lat):
            if latdeg[k]<0:
                l = -l
            if londeg[k]<0:
                lon[k] = -lon[k]
    
    return(lat,lon)

def UTM2latlon(easting,northing,zone,south=False):
    """
    Calculate latitude/longitude from UTM.
    
    Parameters:
        easting: UTM easting (m)
        northing: UTM northing (m)
        zone: UTM zone as integer
        south: Whether UTM zone is from the southern hemisphere
    
    Returns:
        lat: Latitude (decimal degrees)
        lon: Longitude (decimal degrees)
    
    """
    # Set ouptput to lat/lon
    crs = ccrs.Geodetic()
    
    # Convert
    lon,lat = crs.transform_point(
        easting,northing,src_crs=ccrs.UTM(zone=zone,
                                          southern_hemisphere=south))
    
    return(lat,lon)

def latlon2UTM(latitude,longitude,zone,south=False):
    """
    Calculate UTM coordinates from latitude and longitude.

    Parameters:
        latitude: Latitude (decimal degrees)
        longitude: Longitude (decimal degrees)
        zone: UTM zone as integer
        south: Whether UTM zone is from the southern hemisphere
    
    Returns:
        easting: UTM easting (m)
        northing: UTM northing (m)
    """
    
    # Set ouptput to UTM
    crs = ccrs.UTM(zone=zone,southern_hemisphere=south)
    
    # Convert
    easting,northing = crs.transform_point(
        longitude,latitude,src_crs=ccrs.Geodetic())
    
    return(easting,northing)

def pseudom2latlon(easting,northing):
    """
    Calculate latitude/longitude from pseudomercator.
    
    Parameters:
        easting: pseudomercator easting (m)
        northing: pseudomercator northing (m)
    
    Returns:
        lat: Latitude (decimal degrees)
        lon: Longitude (decimal degrees)
    """
    
    # Set ouptput to lat/lon
    crs = ccrs.Geodetic()
    
    # Convert
    lon,lat = crs.transform_point(
        easting,northing,src_crs=ccrs.epsg(3857))
    
    return(lat,lon)
