import numpy as np
from astropy.modeling import models, fitting
from astropy.constants import c as c_light
from PyAstronomy.pyasl import crosscorrRV
from scipy.optimize import curve_fit
from espectro import hdus
import warnings
from pathlib import Path


try:
    from lmfit import Model
    HAS_LMFIT = True
except ImportError:
    HAS_LMFIT = False

if not HAS_LMFIT:
    warnings.warn("lmfit indisponível: ew_gauss devolve ew_err = NaN")

'''
This file contains the functions for the calculation of the equivalent width (EW).
'''

# <script> <wave> <flux> <lim_min_spec> <lim_sup_spec>

#Correction of the spectrum radial velocity
def ccf(wave, flux, w_min=5300, w_max=5600, template="sun_harps_ganymede.fits",
        rmin=-80, rmax=80, margem=3.0, rv_manual=None):

    
    '''Devolve (wave_corrigida, rv_km_s). rv_manual: RV conhecido DESTA estrela (km/s);
    se dado, não usa o template. Sem rv_manual e sem template válido -> erro.'''
    wave, flux = np.asarray(wave).ravel(), np.asarray(flux).ravel()

    if rv_manual is not None:
        rv_ideal = float(rv_manual)
    else:
        tpath = Path(__file__).parent / template        # independente da pasta de execução
        t_wave, t_flux = hdus(str(tpath))               # se falhar, rebenta (de propósito)
        t_wave, t_flux = t_wave.ravel(), t_flux.ravel()

        m  = (wave   > w_min) & (wave   < w_max)
        mt = (t_wave > w_min - margem) & (t_wave < w_max + margem)

        fs  = norm_lin(wave[m],    flux[m],    w_min,          w_max)          - 1.0 #Remove, ou mitiga, o sinal do contínuo
        fts = norm_lin(t_wave[mt], t_flux[mt], w_min - margem, w_max + margem) - 1.0

        rv, Ccf = crosscorrRV(wave[m], fs, t_wave[mt], fts, rmin, rmax, drv=0.5)

        rv_ideal = rv[np.argmax(Ccf)]

        if abs(rv_ideal) > 0.9 * rmax:                  # pico no limite do intervalo -> não fiável
            warnings.warn(f"RV={rv_ideal:.1f} km/s junto ao limite de [{rmin},{rmax}]; "
                          "alarga o intervalo ou verifica o espectro")

    return wave / (1 + rv_ideal / c_light.to("km/s").value), rv_ideal  


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

# Solo line equivalent Line width
def ew_gauss(wave_l, flux_norm):
    """
    Gaussian fit for a normalized isolated absorption line with error estimation using LMFit.

    Parameters:
    - wave_l: Array of wavelengths of the spectral window.
    - flux_norm: Array of normalized flux for the spectral window (continuum at ~1.0)
                 -> obtained from the norm_lin function.

    Returns:
    - ew: Equivalent width value
    - ew_err: Uncertainty in the equivalent width (NaN se não estimável)
    - flux_fit: Array with the modeled flux values (continuum at 1.0).
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
        stdv = max(0.1, (wave_l[-1] - wave_l[0]) / 20)

    # Define Gaussian function for LMFit
    def gaussian(x, amplitude, mean, sigma):
        """amplitude * exp(-0.5 * ((x - mean) / sigma)**2)"""
        return amplitude * np.exp(-0.5 * ((x - mean) / sigma)**2)

    const = np.sqrt(2 * np.pi)

    if HAS_LMFIT:
        # Create LMFit model
        gauss_model = Model(gaussian)

        # Set parameters
        params = gauss_model.make_params(amplitude=amp, mean=mu, sigma=stdv)
        params['amplitude'].min = 0      # Amplitude should be positive
        params['sigma'].min = 1e-10      # Sigma should be positive

        # Perform the fit
        result = gauss_model.fit(flux_inv, params, x=wave_l)

        # Extract parameter values and uncertainties
        amp_fit   = result.params['amplitude'].value
        mu_fit    = result.params['mean'].value
        sigma_fit = result.params['sigma'].value
        amp_err   = result.params['amplitude'].stderr
        sigma_err = result.params['sigma'].stderr

        # EW = amplitude * sigma * sqrt(2*pi) for a Gaussian
        ew = amp_fit * sigma_fit * const

        # Error propagation com covariância A–sigma:
        # Var(EW) = (2*pi) * [ (sigma*sigma_A)^2 + (A*sigma_sigma)^2
        #                      + 2*A*sigma*rho*sigma_A*sigma_sigma ]
        if amp_err is None or sigma_err is None:
            # fit degenerado / parâmetro no bound → LMFit não estimou stderr
            ew_err = np.nan
        else:
            rho = result.params['amplitude'].correl.get('sigma', 0.0)
            var_ew = const**2 * (
                (sigma_fit * amp_err)**2
                + (amp_fit * sigma_err)**2
                + 2.0 * amp_fit * sigma_fit * rho * amp_err * sigma_err
            )
            ew_err = float(np.sqrt(max(var_ew, 0.0)))   # max() protege de var<0 por arredondamento

        # Generate the fitted flux model
        gauss_fit_values = gaussian(wave_l, amp_fit, mu_fit, sigma_fit)
        flux_fit = 1.0 - gauss_fit_values    # Convert back to normalized flux scale

    else:
        # Fallback para astropy quando LMFit não está disponível
        gauss_model = models.Gaussian1D(amplitude=amp, mean=mu, stddev=stdv)
        fit_g = fitting.LevMarLSQFitter()
        gauss_fit = fit_g(gauss_model, wave_l, flux_inv)

        flux_fit = 1.0 - gauss_fit(wave_l)

        amp_fit   = gauss_fit.amplitude.value
        sigma_fit = np.abs(gauss_fit.stddev.value)
        ew = amp_fit * sigma_fit * const
        ew_err = np.nan    # LevMar do astropy não expõe covariância de forma fiável

    return ew, ew_err, flux_fit
