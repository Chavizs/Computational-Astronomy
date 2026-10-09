import numpy as np
from astropy.modeling import models, fitting
from astropy.constants import c as c_light
from PyAstronomy.pyasl import crosscorrRV
from scipy.optimize import curve_fit

'''
This file contains the functions for the calculation of the equivalent width (EW).
'''

# <script> <wave> <flux> <lim_min_spec> <lim_sup_spec>

#Correction of the spectrum radial velocity
def ccf(wave,flux,t_wave,t_flux,rmin=-25,rmax=25):
    '''
    Computes the Cross-Correlation Function (CCF) using the crosscorRV function from PyAstronomy.

    Arguments:
    - wave : array with the original spectrum wavelength
    - flux : array with the original flux
    - t_wave, t_flux : template values of wavelength and flux
    - rm_ : velocities range to use
    
    Returns: the corrected spectrum (based on Doopler effect)
    '''
    #dv = np.linspace(rmin,rmax, int((rmax-rmin)/0.5)) #Velocity range for the CCF, with a step of 0.5 km/s
    rv, Ccf = crosscorrRV(wave,flux, t_wave, t_flux, rmin, rmax, drv=0.5)
    rv_ideal =rv[np.argmax(Ccf)]

    wave = wave(1+rv_ideal/c_light)

    #Returns the corrected wavelenght array
    return wave 

#Flux normalization
def norm_lin(wave, flux, lim_inf_spec, lim_sup_spec,cuts=10):
    '''
    Arguments:
    - wave: wavelength ONLY in the window
    - flux: flux ONLY in the window to normalize

    - Limits of the window containing the risk

    ADD A WINDOW CHECK SIZE
    '''

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
    half_cuts=cuts//2

    for low, high in [(0, half_cuts), (half_cuts, cuts)]:
        while survivor[low:high].sum() > 2:

            worst = low + np.argmax(work[low:high])   #Reves the worst section (with the highest stdv)
            survivor[worst] = False              
            work[worst] = -np.inf #Ensures that will never use the same section

    indexes = np.where(survivor == 1)[0] #Finds the indexes of the sections that survived

    #linear model
    flux_values = np.concatenate([flux_sect[i] for i in indexes]) 
    wave_values = np.concatenate([wave_sect[i] for i in indexes])

    line_coefs = np.polyfit(wave_values, flux_values, 1) 
    flux_linmodel = np.polyval(line_coefs, wave) 

    #returns normalized flux 
    return flux/flux_linmodel

#Solo line equivalent Line width
def ew_gauss(wave_l, flux_norm):
    """
    Gaussian fit for a normalized isolated absorption line.
    
    Parameters:
    - wave_l: Array of wavelengths of the spectal window.
    - flux_norm: Array of normalized flux for the spectral window (continuum at ~1.0) -> obtained from the norm_lin function.
    
    Returns:
    - flux_fit: Array with the modeled flux values (in the original scale, continuum at 1.0).
    """
    # 1. Inverte a linha de absorção para trabalhar com um pico positivo a partir do zero
    flux_inv = 1.0 - flux_norm
    
    # 2. Estimativas iniciais (guesses)
    amp = np.max(flux_inv)
    fwhm_level = amp / 2.0

    # Isola o núcleo da linha para estimar a posição central e a largura
    wave_gauss = wave_l[flux_inv >= fwhm_level]
    
    if len(wave_gauss) > 1:
        mu = np.mean(wave_gauss)
        stdv = np.std(wave_gauss, ddof=1)
    else:
        # Fallback de segurança caso a janela seja muito estreita ou com ruído
        mu = wave_l[np.argmax(flux_inv)]
        stdv = max(0.1 , (wave_l[-1] - wave_l[0]) / 20) 

    # 3. Definição do modelo e ajuste (fitting)
    gauss_model = models.Gaussian1D(amplitude=amp, mean=mu, stddev=stdv)
    fit_g = fitting.LevMarLSQFitter()
    gauss_fit = fit_g(gauss_model, wave_l, flux_inv)

    # plot of the gaussian
    flux_fit = 1- gauss_fit(wave_l)

    amp_fit , stdv_fit= gauss_fit.amplitude.value, np.abs(gauss_fit.stddev.value)
    ew=amp_fit * stdv_fit * np.sqrt(2 * np.pi)  # integral of the Gaussian 
    # Retorna o modelo (para extrair parâmetros) e o array (para desenhar o gráfico)
    return ew , flux_fit

