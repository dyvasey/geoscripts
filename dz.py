"""
Module for processing and plotting detrital zircon data
"""
class DZSample:
    """ Object to hold detrital zircon sample metadata and ages """
    def __init__(self,location,agedata):
        self.location = location
        self.agedata = agedata
