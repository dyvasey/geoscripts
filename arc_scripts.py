"""
Scripts to get data out of ArcMap 10.6. Must be loaded in the ArcMap Python
window.
"""
import json

def lyr_export(path,output):
    """
    Export lyr file to json
    
    Paramters:
        path: Full path to .lyr file
        output: name of output file (.json)
    
    Returns:
        None
    """
    # Make layer object
    lyr = arcpy.mapping.Layer(path)
    
    # Get symbology as json
    text = lyr._arc_object.getsymbology()
    
    # Write json to file
    with open(output,'w') as file:
        json.dump(text,file)
    
    return

