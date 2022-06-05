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

def get_layout_extent(name, map='Map 1', path='./extent.csv'):
    """
    Save layout map extent as wkt polygon in .csv file
    """
    # Get current project
    project = QgsProject.instance()

    # Get layout
    layout = project.layoutManager().layoutByName(name)

    # Get map
    map = layout.itemById(map)
    print(map.crs())

    poly = map.visibleExtentPolygon()

    points = [point for point in poly]

    text = 'POLYGON(('
    for point in points:
        text = text + str(point.x()) + ' ' + str(point.y()) + ','

    text = text +'))'

    df = pd.DataFrame([text],columns=['wkt'])

    df.to_csv(path)
