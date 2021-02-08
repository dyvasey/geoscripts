"""
Module for calculations with Rb-Sr and Sm-Nd isotopic systems
"""
from numpy import exp

def epsNd(Ndrat,Smrat=0,age=0):
    """
    Calculate epsilon Nd using CHUR values from Bouvier et al., 2008
    
    Parameters:
        Ndrat: Measured 143Nd/144Nd ratio
        Smrat: Measured 147Sm/144Nd ratio
        age: Age of sample in Ma for initial calculation
  
    Returns:
        eNd: epsilon Nd for present-day 143Nd/144Nd
        Ndi: initial 143Nd/144Nd ratio
        eNdi: epsilon Nd for initial 143Nd/144Nd ratio
    """
    lambdaSm = 6.54e-12 # Sm-147 decay constant
    years = 10**6 # Converter for Ma to years
    time = age*years # Years for initial calc
    
    Ndi = Ndrat-(Smrat*(exp(lambdaSm*time)-1)) # Calculate initial 143/144
    
    CHUR143 = 0.512630 #CHUR 143Nd/144Nd from Bouvier et al., 2008
    CHUR147 = 0.1960 # CHUR 147Sm/44Nd from Bouvier et al., 2008
    CHUR143i = CHUR143-CHUR147*(exp(lambdaSm*time)-1) # Calculate CHUR for age
    eNd = ((Ndrat/CHUR143)-1)*10**4 # Calculate EpsNd
    eNdi = ((Ndi/CHUR143i)-1)*10**4 # Calculate EpsNdi
    if age==0:   
        return (eNd)
    else:
        return (Ndi,eNdi)

def Srinit(Srrat,Rbrat,age):
    lambdaRb = 1.39e-11 #Rb-87 decay constant
    years = 10**6 #Converter for Ma to years
    time = age*years #years for initial calc
    
    Sri = Srrat-(Rbrat*(exp(lambdaRb*time)-1)) #calculate initial 143/144
    return (Sri)
    
