# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:39:37 2020

@author: dyvas
"""
#Lat/Long decimal degrees from DMS calculator

def dms2deg(latdeg,londeg,latmin,lonmin,latsec=0,lonsec=0):
    lat = (latdeg + (latmin+latsec/60)/60)
    lon = (londeg + (lonmin+lonsec/60)/60)
    return(lat,lon)