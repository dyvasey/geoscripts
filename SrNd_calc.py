# -*- coding: utf-8 -*-

# SrNd Calculator

from numpy import exp
def epsNd(Ndrat,Smrat=0,age=0):
    #Ndrat is measured 143/144 ratio, Smrat is measured 147/44, age in Ma
    lambdaSm = 6.54e-12 #Sm-147 decay constant
    years = 10**6 #Converter for Ma to years
    time = age*years #years for initial calc
    
    Ndi = Ndrat-(Smrat*(exp(lambdaSm*time)-1)) #calculate initial 143/144
    
    CHUR143 = 0.512630 #CHUR 143/144 from Bouvier08
    CHUR147 = 0.1960 #CHUR 147/44 from Bouvier08
    CHUR143i = CHUR143-CHUR147*(exp(lambdaSm*time)-1) #Calculate CHUR for age
    eNd = ((Ndrat/CHUR143)-1)*10**4 #Calculate EpsNd
    eNdi = ((Ndi/CHUR143i)-1)*10**4 #Calculate EpsNdi
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
    
