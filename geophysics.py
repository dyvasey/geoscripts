"""
Functions for doing geophysical calculations
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def viscosity(A,n,d,m,E,P,V,T,strain_rate=1e-15,R=8.31451):
    """
    Calculate viscosity according to equation from ASPECT manual.
    
    For dislocation creep, m = 0. For diffusion creep, n = 1.
    
    Parameters:
        A: Power-law constant
        n: Stress exponent
        d: Grain size (m)
        m: Grain size exponent
        E: Activation energy (J/mol)
        P: Pressure (Pa)
        V: Activation volume (m^3/mol)
        T: Temperature (K)
        strain_rate: square root of the second invariant of 
            the strain rate tensor (s^-1)
        R: Gas constant (J/K*mol)
        
    Returns:
        visc: Viscosity (Pa*s)
    """
    visc = (
        0.5 * A**(-1/n) * d**(m/n) * (strain_rate)**((1-n)/n) * 
        np.exp((E+P*V)/(n*R*T))
        )

    return(visc)

def visc_dislocation(A,n,E,P,V,T,strain_rate=1e-15,R=8.31451):
    """
    Calculate viscosity for dislocation creep (m=0). See viscosity for 
    parameters.
    """
    
    visc = viscosity(A=A,n=n,d=1,m=0,E=E,P=P,V=V,T=T,strain_rate=strain_rate,
                     R=R)
    
    return(visc)

def visc_disl_alt(C,n,t,E,I2=1e-15,R=8.31451):
    
    """
    Alternative method of calculating viscosity for dislocation creep from
    John Naliboff script.
    
    Parameters:
        C: Power-law constant (prefactor)
        n: Stress exponent
        t: Temperature (K)
        E: Activation energy (J/mol)
        I2: Square root of the second invariant of the 
            strain rate tensor (s^-1)
        R: Gas constant (J/K*mol)
    
    Returns:
        v: Viscosity (Pa*s)

    """
    ev = (C**(-1./n)) * (I2**(1./(n))) * np.exp(E/(n*R*(t)))
    
    v = ev/2/I2
    
    return(v)

def visc_diffusion(A,d,m,E,P,V,T,strain_rate=1e-15,R=8.31451):
    """
    Calculate viscosity for diffusion creep (n=1). See viscosity for
    parameters
    """
    
    visc = viscosity(A=A,n=1,d=d,m=m,E=E,P=P,V=V,T=T,strain_rate=strain_rate,
                     R=R)
    
    return(visc)

def visc_composite(visc_dislocation,visc_diffusion):
    """
    Calculate composite diffusion and dislocation creep viscosity.
    
    Parameters:
        visc_dislocation: Viscosity (Pa*s) for dislocation creep
        visc_diffusion: Viscosity (Pa*s) for diffusion creep
    """
    
    visc = (visc_dislocation*visc_diffusion)/(visc_dislocation+visc_diffusion)
    
    return(visc)

def cond_geotherm(thicknesses=[20,20,60],depth=400,
             radiogenic_heat=[1.e-6,2.5e-7,0.],surface_t=273,
             heat_flow=0.05296,thermal_conductivity=2.5):
    """
    Calculate conductive continental geotherm values
    after Chapman86 and Naliboff scripts. Designed to be combined with
    adiabatic geotherm (i.e., asthenosphere temperature set as LAB
    temperature).
    
    Parameters:
        thicknesses: Thicknesses of lithospheric units (km)
        depth: Depth of model (km)
        radiogenic_heat: Radiogenic heat production in each unit (W/m^3)
        surface_t: Surface temperature (K)
        heat_flow: Surface heat flow (W/m^3)
        thermal_conductivity: Thermal conductivity (W/m*K)
    
    Returns:
        temps: Conductive temperatures (K) at each layer boundary
        heat_flows: Heat flows (W/m^3) at each layer boundary
        z: Array of depths (m)
        tc: Conductive temperatures at each depth (K)
    """
    thick_m = [x*1000 for x in thicknesses]
    
    # Set up heat flows and temperatures list
    heat_flows = [heat_flow]
    temps = [surface_t]
    
    for x in range(len(thicknesses)):
    
        # Determine heat flows at each layer boundary
        heat_flows.append(heat_flows[x] - (radiogenic_heat[x]*thick_m[x]))
        
        # Determine temperatures at each layer boundary
        temps.append(
            temps[x] + (heat_flows[x]/thermal_conductivity)*thick_m[x] - 
            (radiogenic_heat[x]*thick_m[x]**2)/(2.*thermal_conductivity)
            )
        
    # Calculate geotherm
    z = np.arange(0,depth+1,1)*1000 # depths in m
    tc = np.zeros(depth+1) # empty array of temperature
    
    # Set boundary locations for slicing tc
    boundaries = [0,thicknesses[0],thicknesses[0]+thicknesses[1],
                  thicknesses[0]+thicknesses[1]+thicknesses[2],
                  depth+1]
    
    # Get each layer as separate depth array
    layers = []
    temp_layers = []
    for x in range(len(thicknesses)+1):
        layers.append(z[boundaries[x]:boundaries[x+1]])
        temp_layers.append(tc[boundaries[x]:boundaries[x+1]])

    # Assign appropriate temperature values to each set of depths
    for x in range(len(layers)-1):
        temp_layers[x] = (
            temps[x] + (heat_flows[x]/thermal_conductivity)*
            (layers[x]-boundaries[x]*1000) - 
            (radiogenic_heat[x]*((layers[x]-boundaries[x]*1000)**2))/
            (2*thermal_conductivity)
            )
    # Assign constant temperature to the asthenosphere before replacing with
    # adiabatic
    temp_layers[-1] = temp_layers[-1] + temps[-1]
    # Combine depths into single array
    tc = np.concatenate(temp_layers)

    
    return(temps,heat_flows,z,tc)

def adiab_geotherm(z,ast=1573,gravity=9.81,thermal_expansivity=2.e-5,
                   heat_capacity=750,depth=400):
    """
    Calculate adiabatic geotherm. Assumes numpy array of depths (z) has
    already been calculated using conc_geotherm()
    
    Parameters:
        z: Array of depths (m)
        ast: Adiabatic surface temperature (K)
        gravity: Gravitational acceleration (m/s^-2)
        thermal_expansivity: Thermal expansivity (K^-1)
        heat_capacity: Heat capacity (J/K*kg)
        depth: Depth of model (km)
    
    Returns:
        ta: Adiabatic temperature for each depth (K)
    """
    ta = np.zeros(depth+1) # empty array for ta
    for i in range(np.size(z)):
      if i==0:
        ta[i] = ast # By design, the adiabatic surface temperature is the LAB temp
      else:
        # See line 124 in source/adiabatic_conditions/compute_profile
        ta[i] = ta[i-1] * (
            1 + (thermal_expansivity * gravity * 1000 * 1./heat_capacity))
        
    return(ta)

def geotherm(thicknesses=[20,20,60],depth=400,
             radiogenic_heat=[1.e-6,2.5e-7,0.],surface_t=273,
             heat_flow=0.05296,thermal_conductivity=2.5,ast=1573,gravity=9.81,
             thermal_expansivity=2.e-5,heat_capacity=750,plot=True,
             save=True):
    """
    Calculate combined conductive and adiabatic geotherm, after Naliboff
    scripts.
    
    Parameters:
        thicknesses: Thicknesses of lithospheric units (km)
        depth: Depth of model (km)
        radiogenic_heat: Radiogenic heat production in each unit (W/m^3)
        surface_t: Surface temperature (K)
        heat_flow: Surface heat flow (W/m^3)
        thermal_conductivity: Thermal conductivity (W/m*K)
        ast: Adiabatic surface temperature (K)
        gravity: Gravitational acceleration (m/s^-2)
        thermal_expansivity: Thermal expansivity (K^-1)
        heat_capacity: Heat capacity (J/K*kg)
        plot: Whether to plot the geotherm
        save: Whether to save boundary temperatures and heat flows to
            separate csv file.
        
    Returns:
        temps: Conductive temperatures (K) at each layer boundary
        heat_flows: Heat flows (W/m^3) at each layer boundary
        z: Array of depths (m)
        tt: Temperature (K) of combined conductive and adiabatic geotherm at
            each depth.
        
    """
    # Conductive geotherm
    temps,heat_flows,z,tc = cond_geotherm(thicknesses=thicknesses,depth=depth,
                       radiogenic_heat=radiogenic_heat,surface_t=surface_t,
                       heat_flow=heat_flow,thermal_conductivity=
                       thermal_conductivity)
    
    # Adiabatic geotherm
    ta = adiab_geotherm(z=z,ast=ast,gravity=gravity,thermal_expansivity=
                        thermal_expansivity,heat_capacity=heat_capacity,
                        depth=depth)
    
    # Combined geotherm
    tt = tc + ta - ast
    
    print('Conductive Boundary Temperatures: ', temps)
    print('Conductive Boundary Heat Flows: ', heat_flows)
    
    print('LAB Depth       = ', sum(thicknesses), 'km')
    print('LAB Temperature = ', tt[sum(thicknesses)], 'K')
    
    if plot==True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(tt,z/1000)
        ax.invert_yaxis()
        ax.set_xlabel('T (K)')
        ax.set_ylabel('Depth (km)')
    
    if save==True:
        output = pd.Series(data=np.concatenate((temps,heat_flows[0:-1]),axis=None),
                           index=['ts1','ts2','ts3','ts4','qs1','qs2','qs3'])
        
        lith = np.sum(thicknesses)
                                     
        output.to_csv('thermal_'+str(lith)+'km.csv')
    
    return(temps,heat_flows,z,tt)

def pressure(z,rho,g=9.81):
    """
    Calculate pressure as a function of depth
    
    Parameters:
        z: depth (m)
        rho: density (kg/m^2)
        g: gravitational accleration (m/s^2)
    
    Returns:
        p: pressure (Pa or kg/m*s^2)
    """
    p = np.zeros(len(z))
    for i in range(1,len(z)):
        p[i] = p[i-1] + rho[i]*g*(z[i]-z[i-1])
    
    return(p)

def density_profile(z,thicknesses=[20,20,60],densities=[2700,2850,3300,3300],
                    depth=400):
    """
    Create numpy array of density for given depth array (z). Currently
    requires upper crust, lower crust, mantle lithosphere, and asthenosphere
    """
    rho = np.zeros(len(z))
    
    # Set boundary locations for slicing
    boundaries = [0,thicknesses[0],thicknesses[0]+thicknesses[1],
                  thicknesses[0]+thicknesses[1]+thicknesses[2],
                  depth+1]
    
    # Get each layer as separate depth array
    dens_layers = []
    for x in range(len(densities)):
        dens_layers.append(rho[boundaries[x]:boundaries[x+1]])
        dens_layers[x] = dens_layers[x] + densities[x]
    
    # Combine into single array
    rho = np.concatenate(dens_layers)
    
    return(rho)

def viscosity_profile(A,A_df,n,d,m,E,E_df,V,V_df,thicknesses=[20,20,60],
                      densities=[2700,2850,3300,3300],heat_flow=0.05296,
                    depth=400,strain_rate=1e-15,R=8.31451,plot=True):
    """
    Calculate composite viscosity profile using factors as reported to ASPECT.
    """
    # Calculate geotherm
    temps,heat_flows,z,tt = geotherm(thicknesses=thicknesses,depth=depth,
                                     heat_flow=heat_flow,plot=plot)
    
    # Assign densities to array
    rho = density_profile(z=z,thicknesses=thicknesses,densities=densities,
                          depth=depth)
    
    # Calculate pressure
    p = pressure(z,rho)
    
    # Set boundary locations for slicing
    boundaries = [0,thicknesses[0],thicknesses[0]+thicknesses[1],
              thicknesses[0]+thicknesses[1]+thicknesses[2],
              depth+1]
    
    # Calculate dislocation and diffusion creep for each layer
    disl = np.zeros(len(z))
    diff = np.zeros(len(z))
    #alt = np.zeros(len(z))
    disl_layers = []
    diff_layers = []
    #alt_layers = []
    for x in range(len(A)):
        # Get slices for each layer
        disl_layers.append(disl[boundaries[x]:boundaries[x+1]])
        diff_layers.append(diff[boundaries[x]:boundaries[x+1]])
        #alt_layers.append(alt[boundaries[x]:boundaries[x+1]])
        # Slice pressure and temperature arrays
        p_slice = p[boundaries[x]:boundaries[x+1]]
        t_slice = tt[boundaries[x]:boundaries[x+1]]
        # Calculate viscosities for each slice
        disl_layers[x] = visc_dislocation(A=A[x],n=n[x],E=E[x],P=p_slice,
                                          V=V[x],T=t_slice)
        
        diff_layers[x] = visc_diffusion(A=A_df[x],m=m[x],d=d,E=E_df[x],P=p_slice,
                                        V=V_df[x],T=t_slice)
        
        #alt_layers[x] = visc_disl_alt(C=A[x],n=n[x],E=E[x],t=t_slice)
        
    
    # Recombine slices into single arrays
    disl = np.concatenate(disl_layers)
    diff = np.concatenate(diff_layers)
    #alt = np.concatenate(alt_layers)
    
    
    # Calculate composite creep
    comp = visc_composite(disl,diff)
    
    # Plot Profile
    if plot==True:
        fig = plt.figure(dpi=300)
        viscosities = [disl,diff,comp]
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        axes = [ax1,ax2,ax3]
        titles = ['Dislocation Creep','Diffusion Creep','Composite']
        
        for x in range(3): 
            axes[x].plot(viscosities[x], z/1e3, linestyle='-',  color='blue')
            axes[x].invert_yaxis();
            axes[x].set_xlabel('Viscosity (Pa*s)');
            axes[x].set_ylabel('Depth (km)');
            axes[x].set_xlim([1e16,1e30])
            axes[x].set_ylim([depth,0])
            axes[x].set_xscale('log')
            axes[x].set_title(titles[x])
            
        plt.tight_layout()
    
    return(z,comp,disl,diff,tt)
    
    
    
    