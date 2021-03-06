"""
Module for processing and plotting detrital zircon data
"""
import seaborn as sns

class DZSample:
    """ Object to hold detrital zircon sample metadata and ages """
    
    # Define basic attributes
    def __init__(self,name,location=None,agedata=None):
        self.name = name
        self.location = location
        self.agedata = agedata

    def kde(self,**kwargs):
        sns.kdeplot(self.agedata['238 Age'],log_scale=True,**kwargs)