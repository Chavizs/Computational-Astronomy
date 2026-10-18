from EW import *
from espectro import hdus
import warnings
import pandas as pd
from evaluate_racios import aplicar_calibracao, combinar_estimativas 



def estimar_teff(fits_path, cal_csv="Calibration.csv", cv_csv="Calibration_CV.csv", nsigma=2.5):
    '''
    Estima a Teff de uma estrela (SEM ESTAR EM REPOUSO) com a calibração.

    1) mede as EW das riscas do Calibration.csv (mesmas janelas da calibração);
    2) calcula os mesmos rácios EW_1/EW_2 e, para cada um, Teff = polinómio(rácio);
    3) rejeita estimativas discrepantes (mediana + MAD, nsigma) e combina (média pesada 1/sigma^2).

    Devolve (res, est): res = dict(Teff, sigma, n_antes, n_depois);
                        est = DataFrame com o valor do rácio e a Teff de cada rácio.
    '''
    cv = pd.read_csv(cv_csv).iloc[0]     # se faltar, dá erro: corre validar_calibracao
    #Loads the data to be used: Calibration and the data in the .fist file 
    cal = pd.read_csv(cal_csv)
    wave, flux = hdus(fits_path)
    
    #Corrects the radial velocity
    wave, rv = ccf(wave, flux) 

    wave, flux = np.ravel(wave).astype(float), np.ravel(flux).astype(float)

    # 1) EW de cada risca, medida uma só vez (mesma lógica do auto_measure)
    
    janelas = {}
    #Getting the range used during the calibation to perform the gaussian fit
    for _, c in cal.iterrows():
        janelas[round(c["line_1"], 2)] = c["rng_1"]
        janelas[round(c["line_2"], 2)] = c["rng_2"]

    #EW's computation
    ews, errs = {}, {}
    for line, rng in janelas.items():
        lim_inf, lim_sup = line - rng / 2, line + rng / 2
        m = (wave > lim_inf) & (wave < lim_sup)
        try:
            flux_n = norm_lin(wave[m], flux[m], lim_inf, lim_sup)
            #Discards bad fits
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = ew_gauss(wave[m], flux_n)
                
      # aceita ew ou (ew, flux_fit)
            ew, ew_err = out[0], out[1]
            fit_warn = [wi for wi in w if issubclass(wi.category, (RuntimeWarning, UserWarning))
                        and "deprecat" not in str(wi.message).lower()]
            bom = (np.isfinite(ew) and np.isfinite(ew_err)
                and 1e-3 < ew <= 1 and ew_err > 0 and not fit_warn)
            ews[line]  = ew     if bom else np.nan
            errs[line] = ew_err if bom else np.nan

        except Exception:
            ews[line] = errs[line] = np.nan

    # 2) Ratio and Teff for each fit in the calibration (sem extrapolar fora do intervalo calibrado)
    linhas = []
    for _, c in cal.iterrows():
        l1, l2 = round(c["line_1"], 2), round(c["line_2"], 2)
        ew1, ew2, e1, e2 = ews[l1], ews[l2], errs[l1], errs[l2]
        r = ew1 / ew2 if (ew1 > 0 and ew2 > 0) else np.nan
        T = np.nan
        if np.isfinite(r) and c["ratio_min"] <= r <= c["ratio_max"]:
            T = aplicar_calibracao(r, c)

        linhas.append({"ratio": c["ratio"], "valor_racio": r, "Teff": T,
                       "sigma_cal": c["scatter_K"]})
        
    est = pd.DataFrame(linhas)

    # 3) rejeição de discrepantes e combinação
    T, sig = est["Teff"].to_numpy(float), est["sigma_cal"].to_numpy(float)
    out = combinar_estimativas(T, sig, nsigma)
    if out is None:
        raise RuntimeError("Menos de 3 rácios com estimativa válida.")
    teff, disp, keep = out
    est["usado"] = keep
    sigma = float(cv["rms_K"] * max(1.0, disp / cv["disp_med_K"]))
    return dict(Teff=teff, sigma=sigma, sigma_scatter=disp, rv=float(rv),
                n_antes=int(np.isfinite(T).sum()), n_depois=int(keep.sum())), est



