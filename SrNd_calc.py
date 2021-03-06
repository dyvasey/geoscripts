"""
Module for calculations with Rb-Sr and Sm-Nd isotopic systems
"""
from numpy import exp
import pandas as pd

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
    
    Ndi = Ndrat-(Smrat*(exp(lambdaSm*time)-1)) # Calculate initial 143Nd/144Nd
    
    CHUR143 = 0.512630 #CHUR 143Nd/144Nd from Bouvier et al., 2008
    CHUR147 = 0.1960 # CHUR 147Sm/44Nd from Bouvier et al., 2008
    CHUR143i = CHUR143-CHUR147*(exp(lambdaSm*time)-1) # Calculate CHUR for age
    eNd = ((Ndrat/CHUR143)-1)*10**4 # Calculate EpsNd
    eNdi = ((Ndi/CHUR143i)-1)*10**4 # Calculate EpsNdi
    # If age a single integer, create iterable list.
    if isinstance(age,int):
        age = pd.Series(age)
        
    if 0 not in age.values: 
        return (Ndi,eNdi)
    elif 0 in age.values:          
        return (eNd)
    else:
        print('Mix of zero and non-zero ages')

def Srinit(Srrat,Rbrat,age):
    """
    Calculate initial 87Sr/86Sr.
    
    Paramters:
        Srrat: Measured 87Sr/86Sr ratio
        Rbrat: Measured 87Rb/86Sr ratio
        age: Age of sample in Ma
    
    Returns:
        Sri: initial 87Sr/86Sr ratio
    """
    
    lambdaRb = 1.39e-11 # Rb-87 decay constant
    years = 10**6 # Converter for Ma to years
    time = age*years # Years for initial calc
    
    Sri = Srrat-(Rbrat*(exp(lambdaRb*time)-1)) # Calculate initial 87Sr/86Sr
    return (Sri)

def RbSr_rat(Rb,Sr,Srrat):
    """
    Calculate 87Rb/86Sr from Rb and Sr concentrations
    
    Paramters:
        Rb: Rb concentration (ppm)
        Sr: Sr concentration (ppm)
        Srrat: 87Sr/86Sr ratio
    
    Returns:
        rbsr8786: Calculated 87Rb/86Sr ratio
        
    """
    # Fundamental Rb and Sr isotopic parameters - from CIAAW
    Rb85_mass = 84.91178974 # Da
    Rb87_mass = 86.90918053 # Da
    
    Rb85_abund = 0.7217
    Rb87_abund = 0.2783
    

    Sr84_mass = 83.913419
    Sr86_mass = 85.90926073
    Sr87_mass = 86.90887750
    Sr88_mass = 87.90561226
    
    # Sr abundances vary - only used for calculation of ratios that 
    # don't vary
    Sr84_abund = 0.0056
    Sr86_abund = 0.0986
    Sr87_abund = 0.0700 # Not used
    Sr88_abund = 0.8258
    
    Sr_8886 = Sr88_abund/Sr86_abund # 88Sr/86Sr ratio - doesn't vary
    Sr_8486 = Sr84_abund/Sr86_abund # 84Sr/86Sr ratio - doesn't vary
    
    # Calculate true abundances
    Sr86_abund_calc = Srrat/(Srrat+Sr_8886+Sr_8486+1)
    Sr84_abund_calc = Sr86_abund_calc*Sr_8486
    Sr88_abund_calc = Sr86_abund_calc*Sr_8886
    Sr87_abund_calc = Sr86_abund_calc*Srrat
    
    # Total Mass for Rb and Sr
    Rb_mass = Rb85_mass*Rb85_abund + Rb87_mass*Rb87_abund

    Sr_mass = (
        Sr84_mass*Sr84_abund_calc + Sr86_mass*Sr86_abund_calc
        + Sr87_mass*Sr87_abund_calc + Sr88_mass*Sr88_abund_calc
        )
    
    # 87Rb and 86Sr
    Rb87 = Rb*Rb87_abund/Rb_mass # Get mol of Rb87
    Sr86 = Sr*Sr86_abund_calc/Sr_mass # Get mol of Sr86
    rbsr8786 = Rb87/Sr86
    check = (Rb/Sr)*(2.69295 + 0.28304*Srrat)
    print('Check: ',check)
    
    return(rbsr8786)
    
def SmNd_rat(Sm,Nd,Ndrat):    
    """
    Calculate 147Sm/144Nd using reported Sm, Nd, and 143Nd/144Nd
    
    Parameters:
        Sm: Sm concentration (ppm)
        Nd: Nd concentration (ppm)
        Ndrat: 143Nd/144Nd ratio
    
    Returns:
        smnd147/144: Calculated 147Sm/144Nd ratio
    """
    
    # Sm and Nd isotopic parameters - from CIAAW
    Sm144_mass = 143.91201
    Sm147_mass = 146.91490
    Sm148_mass = 147.91483
    Sm149_mass = 148.917191
    Sm150_mass = 149.917282
    Sm152_mass = 151.919739
    Sm154_mass = 153.92222
    
    Sm144_abund = 0.0308
    Sm147_abund = 0.1500
    Sm148_abund = 0.1125
    Sm149_abund = 0.1382
    Sm150_abund = 0.0737
    Sm152_abund = 0.2674
    Sm154_abund = 0.2274
    
    Nd142_mass = 141.90773
    Nd143_mass = 142.90982
    Nd144_mass = 143.91009
    Nd145_mass = 144.91258
    Nd146_mass = 145.91312
    Nd148_mass = 147.91690
    Nd150_mass = 149.920902
    

    # Nd abundances vary 
    
    # Non-varying Nd ratios (from Faure et al., 2005)
    Nd_146144 = 0.7219 # Hamilton et al., 1983
    Nd_142144 = 1.141827
    Nd_145144 = 0.348417
    Nd_148144 = 0.241578
    Nd_150144 = 0.236418
    
    # Calculate Nd abundances
    Nd144_abund = Ndrat/(Ndrat + 1 + Nd_146144 + Nd_142144 + Nd_145144
                         + Nd_148144 + Nd_150144)
    Nd142_abund = Nd_142144*Nd144_abund
    Nd143_abund = Ndrat*Nd144_abund
    Nd145_abund = Nd_145144*Nd144_abund
    Nd146_abund = Nd_146144*Nd144_abund   
    Nd148_abund = Nd_148144*Nd144_abund
    Nd150_abund = Nd_150144*Nd144_abund
    
    # Total mass for Sm and Nd
    Sm_mass = (
        Sm144_mass*Sm144_abund + Sm147_mass*Sm147_abund + Sm148_mass*
        Sm148_abund + Sm149_mass*Sm149_abund + Sm150_mass*Sm150_abund +
        Sm152_mass*Sm152_abund + Sm154_mass*Sm154_abund
        )
    
    Nd_mass = (
        Nd142_mass*Nd142_abund + Nd143_mass*Nd143_abund + 
        Nd144_mass*Nd144_abund + Nd145_mass*Nd145_abund +
        Nd146_mass*Nd146_abund + Nd148_mass*Nd148_abund +
        Nd150_mass*Nd150_abund
        )
    
    # 147Sm and 143Nd
    Sm147 = Sm*Sm147_abund/Sm_mass
    Nd144 = Nd*Nd144_abund/Nd_mass
    
    smnd147144 = Sm147/Nd144
    
    check1 = (Sm/Nd) * (0.53149+0.14252*Ndrat)
    check2 = (Sm/Nd) * 0.602
    
    print('Check1: ',check1)
    print('Check2: ',check2)
    
    return(smnd147144)