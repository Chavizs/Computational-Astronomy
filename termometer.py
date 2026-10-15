def estimar_teff(fits_path, cal_csv="Calibration.csv", nsigma=2.5):
    '''
    Estima a Teff de uma estrela (espectro já em repouso, p.ex. *_rv.fits) com a calibração.

    1) mede as EW das riscas do Calibration.csv (mesmas janelas da calibração);
    2) calcula os mesmos rácios EW_1/EW_2 e, para cada um, Teff = polinómio(rácio);
    3) rejeita estimativas discrepantes (mediana + MAD, nsigma) e combina (média pesada 1/sigma^2).

    Devolve (res, est): res = dict(Teff, sigma, n_antes, n_depois);
                        est = DataFrame com o valor do rácio e a Teff de cada rácio.
    '''
    #Loads the data to be used: Calibration and the data in the .fist file 
    cal = pd.read_csv(cal_csv)
    wave, flux = hdus(fits_path)
    wave, flux = np.ravel(wave).astype(float), np.ravel(flux).astype(float)

    # 1) EW de cada risca, medida uma só vez (mesma lógica do auto_measure)
    
    janelas = {}
    #Getting the range used during the calibation to perform the gaussian fit
    for _, c in cal.iterrows():
        janelas[round(c["line_1"], 2)] = c["rng_1"]
        janelas[round(c["line_2"], 2)] = c["rng_2"]

    #EW's computation
    ews = {}
    for line, rng in janelas.items():
        lim_inf, lim_sup = line - rng / 2, line + rng / 2
        m = (wave > lim_inf) & (wave < lim_sup)
        try:
            flux_n = norm_lin(wave[m], flux[m], lim_inf, lim_sup)
            #Discards bad fits
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = ew_gauss(wave[m], flux_n)
                
            ew = out[0] if isinstance(out, tuple) else out          # aceita ew ou (ew, flux_fit)
            fit_warn = [wi for wi in w if issubclass(wi.category, (RuntimeWarning, UserWarning))
                        and "deprecat" not in str(wi.message).lower()]
            ews[line] = ew if (np.isfinite(ew) and 0 < ew <= 1 and not fit_warn) else np.nan
            
        except Exception:
            ews[line] = np.nan

    # 2) Ratio and Teff for each fit in the calibration (sem extrapolar fora do intervalo calibrado)
    linhas = []
    for _, c in cal.iterrows():
        ew1, ew2 = ews[round(c["line_1"], 2)], ews[round(c["line_2"], 2)]
        r = ew1 / ew2 if (ew1 > 0 and ew2 > 0) else np.nan
        T = np.nan
        if np.isfinite(r) and c["ratio_min"] <= r <= c["ratio_max"]:
            T = np.polyval([c[f"c{i}"] for i in range(int(c["grau"]), -1, -1)], r)
        linhas.append({"ratio": c["ratio"], "valor_racio": r, "Teff": T, "sigma_cal": c["scatter_K"]})
    est = pd.DataFrame(linhas)

    # 3) rejeição de discrepantes e combinação
    T, sig = est["Teff"].to_numpy(float), est["sigma_cal"].to_numpy(float)
    ok = np.isfinite(T)
    if ok.sum() < 3:
        raise RuntimeError(f"Só {ok.sum()} rácios com estimativa válida.")
    keep = ok.copy()
    for _ in range(20):
        med = np.median(T[keep])
        s = max(1.4826 * np.median(np.abs(T[keep] - med)), np.median(sig[keep]))
        novo = ok & (np.abs(T - med) <= nsigma * s)
        if (novo == keep).all():
            break
        keep = novo
    n = int(keep.sum())
    w_ = 1 / sig[keep] ** 2
    teff = np.sum(w_ * T[keep]) / np.sum(w_)
    disp = np.sqrt(np.sum(w_ * (T[keep] - teff) ** 2) / np.sum(w_) * n / max(n - 1, 1))
    est["usado"] = keep
    return dict(Teff=float(teff), sigma=float(disp), n_antes=int(ok.sum()), n_depois=n), est
