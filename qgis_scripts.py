"""
Scripts for use in QGIS

Functions need to be run from QGIS console as currently written
"""
import pandas as pd
import os

def get_colors(filename='colors.csv'):
    """
    Get colors for active layer in QGIS
    
    Parameters:
        filename: Name of file to ouput colors to
        
    Returns:
        names: Values in the field of interest.
        colors: Colors corresponding to the values.
    """
    # Get active layer
    layer = iface.activeLayer()

    # Get to the symbology
    renderer = layer.renderer()
    categories = renderer.categories()

    # Get name and corresponding color
    names = []
    colors = []
    for cat in categories:
        properties = cat.symbol().symbolLayer(0).properties()
        colors.append(properties['color'])
        names.append(cat.value())
    
    # Put into a Pandas series
    output = pd.Series(data=colors,index=names)
    
    # Get working path for QGIS project
    working_path = QgsProject.instance().readPath("./")
    csv_path = os.path.join(working_path,filename)
    
    # Write CSV
    output.to_csv(csv_path)

    return(names,colors)
