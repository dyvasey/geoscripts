# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 09:58:43 2021

@author: dyvas
"""
from shapely.geometry import Polygon
from shapely.geometry import Point
import numpy as np
import pandas as pd
import math
    

def classifyTAS(df):
    
    b = Polygon([(45,0),(52,0),(52,5),(45,5)])
    ba = Polygon([(52,0),(57,0),(57,5.9),(52,5)])
    bta = Polygon([(52,5),(57,5.9),(53,9.3),(49.4,7.3)])
    tb = Polygon([(45,5),(52,5),(49.4,7.3)])
    tepbsn = Polygon([(41,3),(45,3),(45,5),(49.4,7.3),(45,9.4),(41,7)])
    
    df['alkalis'] = df.K2O + df.Na2O
    
    #convert coordinates to tuples for multipoint
    tup = list(df[['SiO2','alkalis']].itertuples(index=True,name=None))
    for x in tup:
        i = x[0]
        if pd.isna(df.loc[i,'Lithology']):              
            point = Point((x[1],x[2]))           
            if b.contains(point):
                df.loc[i,'Lithology']='Basalt'
            elif ba.contains(point):
                df.loc[i,'Lithology']='Basaltic Andesite'
            elif bta.contains(point):
                df.loc[i,'Lithology']='Basaltic Trachyandesite'
            elif tb.contains(point):
                df.loc[i,'Lithology']='Trachybasalt'
            elif tepbsn.contains(point):
                df.loc[i,'Lithology']='Tephrite' #missing basanite
        else:
            print(i)
    return(df)
    

    
    #line1 = [[30,41,41,45,48.4,52.5,30],[0,0,7,9.4,11.5,14,24.15]] 
    #line2 = [[41,45,45,41],[0,0,3,3]]
    #line3 = [[45,52,52,45],[0,0,5,5]] #basalt
    #line4 = [[52,57,57,52],[0,0,5.9,5]] #basaltic adnesite
    #line5 = [[57,63,63,57],[0,0,7,5.9]]
    #line6 = [[63,77,69,63],[0,0,8,7]]
    #line7 = [[77,100,100,69,69],[0,0,25,25,8]]
    #line8 = [[45,52,49.4],[5,5,7.3]] #trachybasalt
    #line9 = [[52,57,53,49.4],[5,5.9,9.3,7.3]] #basaltic trachyandesite
    #line10 = [[57,63,57.6,53],[5.9,7,11.7,9.3]]
    #line11 = [[63,69,69,57.6],[7,8,17.73,11.7]]
    #line12 = [[41,45,45,49.4,45,41],[3,3,5,7.3,9.4,7]] #teph/basanite
    #line13 = [[49.4,53,48.4,45],[7.3,9.3,11.5,9.4]]
    #line14 = [[53,57.6,52.5,48.4],[9.3,11.7,14,11.5]]
    #line15 = [[57.6,69,30],[11.7,17.73,24.15]]
    #lines = [line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11,
    #             line12,line13,line14,line15]