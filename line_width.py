import numpy as np
from astropy.modeling import models, fitting

def ew_gauss(wave_l, flux_norm):
    """
    Gaussian fit for a normalized isolated absorption line.
    
    Parameters:
    - wave_l: Array of wavelengths of the spectal window.
    - flux_norm: Array of normalized flux for the spectral window (continuum at ~1.0) -> obtained from the norm_lin function.
    
    Returns:
    - flux_fit: Array with the modeled flux values (in the original scale, continuum at 1.0).
    """
    # Inverts the flux values (so the peak is upwords)
    flux_inv = 1.0 - flux_norm
    
    # Conditions for Inicial Guesses
    amp = np.max(flux_inv)
    fwhm_level = amp / 2.0
    wave_gauss = wave_l[flux_inv >= fwhm_level]
    
    if len(wave_gauss) > 1:
        mu = np.mean(wave_gauss)
        stdv = np.std(wave_gauss, ddof=1)
    else:
        # Fallback in case window is way too small or contains too much noise
        mu = wave_l[np.argmax(flux_inv)]
        stdv = max(0.1 , (wave_l[-1] - wave_l[0]) / 20) 

    # model fit -- Gaussian 
    gauss_model = models.Gaussian1D(amplitude=amp, mean=mu, stddev=stdv)
    fit_g = fitting.LevMarLSQFitter()
    gauss_fit = fit_g(gauss_model, wave_l, flux_inv)
    
    #intrgal calculation of W_lambda := ew
    amp_fit , stdv_fit= gauss_fit.amplitude.value, gauss_fit.stddev.value
    ew=amp_fit * stdv_fit * np.sqrt(2 * np.pi)   

    return ew
