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

    # 4. Reconverte a gaussiana ajustada para o fluxo normalizado (com contínuo em 1.0)
    flux_fit = 1.0 - gauss_fit(wave_l)

    # Retorna o modelo (para extrair parâmetros) e o array (para desenhar o gráfico)
    return flux_fit