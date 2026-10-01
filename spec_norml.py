
import numpy as np 
import sys

# <script> <wave> <flux> <lim_min_spec> <lim_sup_spec>


def norm_lin(wave, flux, lim_inf_spec, lim_sup_spec):
    lim_inf_spec=int(lim_inf_spec)
    lim_sup_spec=int(lim_sup_spec)
    
    size= lim_sup_spec-lim_inf_spec

    cuts=10
    step=int(size/cuts)

    stdvs=np.empty(cuts)
    flux_sect = np.empty(cuts) #Array with the flux values of each section of the spectrum
    wave_sect = np.empty(cuts) #Array with the wavelength values of each section of the spectrum

    #Creates array with the flux standerd deviation values for each section of the spectrum
    for i in range(1,cuts+1):
        flux_sect[i-1] = flux[lim_inf_spec+(i-1)*step : lim_inf_spec + i*step]
        wave_sect[i-1] = wave[lim_inf_spec+(i-1)*step : lim_inf_spec + i*step]

        stdvs[i-1] = np.std(flux_sect[i-1])


    survivor=np.ones(cuts) #Masks the interesting intervals
    while True:

        too_far=np.argmax(stdvs) #Finds the section with the highest standard deviation
        survivor[too_far] = 0 #Masks the section with the highest standard deviation
        if np.sum(survivor) < 3:
            break

    indexes = np.where(survivor == 1)[0] #Finds the indexes of the sections that survived

    flux_values = np.concatenate(flux_sect[indexes]) #Finds the flux values of the sections that survived. To be used to fit the linear model
    wave_values = np.concatenate(wave_sect[indexes]) #Finds the wavelength values of the sections that survived. To be used to fit the linear model

    line_coefs = np.polyfit(wave_values, flux_values, 1) 
    flux_linmodel = np.polyval(line_coefs, wave) #Finds the linear model of the flux values

    return flux/flux_linmodel





if __name__ == "__main__":
    norm_lin()
