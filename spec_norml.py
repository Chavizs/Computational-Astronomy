import numpy as np 
'''
This funnction automatically normalizes the flux in a specific spectal zone using a linear regression.
Function stucture: <wave> <flux> <lim_min_spec> <lim_sup_spec> <cuts:Optional>
Up to now, it onlies uses 3 sections out of the #cuts passed by the user/ set to default.
The reason of using 3 sections resides in the fact of trying to maximize the chances of having two section on different ends of
the spectral zone chosen
'''
def norm_lin(wave, flux, lim_inf_spec, lim_sup_spec,cuts=10):

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
            mask=(wave >= edges[i]) & (wave <= edges[i+1])

        flux_values=flux[mask]
        flux_sect.append(flux_values)
        wave_sect.append(wave[mask])

        stdvs[i] = np.std(flux_values) #Calculates the standard deviation of the flux values in the section

    survivor=np.ones(cuts) #Masks the interesting intervals
    work=stdvs.copy() #Creates a copy of the stdvs array to work with (allows to keep the original stdvs in case of study the stdvs of sections)

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
    flux_linmodel = np.polyval(line_coefs, wave) 

    return flux/flux_linmodel
