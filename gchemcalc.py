"""
Module for geochemical calculations
"""
from shapely.geometry import Polygon
from shapely.geometry import Point
import pandas as pd
    

def classifyTAS(df):
    """
    Classify samples missing lithology using TAS diagram.
    
    Uses values for TAS diagram of Le Bas et al., 1986 as compiled in source
    code of GCDkit (Janousek et al., 2006). Only a subset of TAS fields are
    currently implemented.
    
    Parameters:
        df: Pandas dataframe with SiO2, K2O, and Na2O values and 'Lithology'
            column
    
    Returns:
        df: Pandas dataframe with 'Lithology' updated
    
    """
    
    # Create Shapely polygons for TAS fields
    b = Polygon([(45,0),(52,0),(52,5),(45,5)])
    ba = Polygon([(52,0),(57,0),(57,5.9),(52,5)])
    bta = Polygon([(52,5),(57,5.9),(53,9.3),(49.4,7.3)])
    tb = Polygon([(45,5),(52,5),(49.4,7.3)])
    tepbsn = Polygon([(41,3),(45,3),(45,5),(49.4,7.3),(45,9.4),(41,7)])
    ptep = Polygon([(45,9.4),(48.4,11.5),(53,9.3),(49.4,7.3)])
    ta = Polygon([(57,5.9),(49.4,7.3),(57.6,11.7),(63,7)])
    
    # Calculate total alkalis
    df['alkalis'] = df.K2O + df.Na2O
    
    # Convert coordinates to tuples for Shapely points
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
                df.loc[i,'Lithology']='Tephrite/Basanite'
            elif ptep.contains(point):
                df.loc[i,'Lithology']='Phono-tephrite' 
            elif ta.contains(point):
                df.loc[i,'Lithology']='Trachyandesite' 
    
    return(df)
