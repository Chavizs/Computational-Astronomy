import numpy as np 

# <script> <wave> <flux> <lim_min_spec> <lim_sup_spec>

def norm_lin(wave, flux, lim_inf_spec, lim_sup_spec):

    edges=np.linspace(lim_inf_spec,lim_sup_spec,cuts+1) #Creates the steps 

    stdvs = np.full(cuts, np.nan)
    #Empty lists to store the values of each section
    flux_sect = [] 
    wave_sect =[] 
    
    #Creates array with the flux standerd deviation values for each section of the spectrum
    for i in range(0,cuts):
        if i < cuts-1:
            mask=(wave >= edges[i]) & (wave < edges[i+1])
        else:
            mask=(wave > edges[i]) & (wave < edges[i+1])

        flux_values=flux[mask]
        flux_sect.append(flux_values)
        wave_sect.append(wave[mask])

        stdvs[i] = np.std(flux_values) #Calculates the standard deviation of the flux values in the section
        
    survivor=np.ones(cuts) #Masks the interesting intervals
    work=stdvs.copy()
    
    while True:

        too_far=np.argmax(work) 
        survivor[too_far] = 0
        work[too_far]=0
        if np.sum(survivor) <= 3:
            break

    indexes = np.where(survivor == 1)[0] #Finds the indexes of the sections that survived

    #linear model
    flux_values = np.concatenate([flux_sect[i] for i in indexes]) 
    wave_values = np.concatenate([wave_sect[i] for i in indexes])

    line_coefs = np.polyfit(wave_values, flux_values, 1) 
    flux_linmodel = np.polyval(line_coefs, wave) #Finds the linear model of the flux values

    return flux/flux_linmodel
