#!/usr/bin/env python3
"""
STEEW - Teff a partir de rácios de larguras equivalentes (EW).

Uso (terminal):
    python STEEW.py espectro.fits                 # RV pelo template (CCF)
    python STEEW.py espectro.fits --rv -15.9      # RV conhecido (km/s), sem template
    python STEEW.py espectro.fits --detalhe       # mostra também os rácios usados

Uso (Python):
    from STEEW import estimar_teff
    res, est = estimar_teff("espectro.fits")      # res["Teff"], res["sigma"], ...

Ficheiros na MESMA pasta que este script:
    Calibration.csv            (build_calibration)
    Calibration_CV.csv         (validar_calibracao)  -> dá o erro do Teff
    sun_harps_ganymede.fits    (template, só para o CCF; dispensável com --rv)

Dependências: numpy, pandas, astropy, lmfit, PyAstronomy
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.constants import c as c_light
from lmfit import Model
from PyAstronomy.pyasl import crosscorrRV

try:
    HERE = Path(__file__).resolve().parent
except NameError:                                   # Jupyter / execução interativa
    HERE = Path.cwd()

C_KMS = c_light.to("km/s").value


# ----------------------------------------------------------------------------
# Leitura do espectro
# ----------------------------------------------------------------------------
def hdus(file_path):
    '''Devolve (wave, flux) em Å, como arrays 1D. Se existir WAVE_AIR (ex.: ESPRESSO,
    onde WAVE é vácuo) usa-o: as listas de riscas estão em comprimentos de onda no ar.'''
    with fits.open(file_path) as hduls:
        
        if len(hduls) == 1:                         # só PrimaryHDU com fluxo + WCS
            flux = hduls[0].data.astype(float)
            hdr = hduls[0].header
            wave = hdr["CRVAL1"] + (np.arange(flux.size) + 1 - hdr["CRPIX1"]) * hdr["CDELT1"]
        else:                                       # tabela (Phase 3)
            data = fits.getdata(file_path)
            names = [n.upper() for n in data.columns.names]
            flux = np.array(data["FLUX"], dtype=float)
            for col in ("WAVE_AIR", "WAVE", "WAVELENGTH"):
                if col in names:
                    wave = np.array(data[col], dtype=float)
                    break
            else:
                raise KeyError(f"Sem coluna de comprimento de onda em {file_path}: {names}")
    return np.ravel(wave), np.ravel(flux)


# ----------------------------------------------------------------------------
# Normalização do contínuo (linear, nas secções menos ruidosas da janela)
# ----------------------------------------------------------------------------
def norm_lin(wave, flux, lim_inf_spec, lim_sup_spec, cuts=10):
    '''wave, flux: só a janela. lim_*: limites da janela. Devolve o fluxo normalizado.'''
    edges = np.linspace(lim_inf_spec, lim_sup_spec, cuts + 1)
    stdvs = np.full(cuts, np.nan)
    flux_sect, wave_sect = [], []

    for i in range(cuts):
        if i < cuts - 1:
            mask = (wave >= edges[i]) & (wave < edges[i + 1])
        else:
            mask = (wave >= edges[i]) & (wave <= edges[i + 1])
        flux_sect.append(flux[mask])
        wave_sect.append(wave[mask])
        stdvs[i] = np.std(flux[mask])

    survivor = np.ones(cuts)
    work = stdvs.copy()
    half = cuts // 2
    for low, high in [(0, half), (half, cuts)]:     # em cada metade ficam as 2 secções mais calmas
        while survivor[low:high].sum() > 2:
            worst = low + np.argmax(work[low:high])
            survivor[worst] = False
            work[worst] = -np.inf

    idx = np.where(survivor == 1)[0]
    flux_values = np.concatenate([flux_sect[i] for i in idx])
    wave_values = np.concatenate([wave_sect[i] for i in idx])
    coefs = np.polyfit(wave_values, flux_values, 1)
    return flux / np.polyval(coefs, wave)


# ----------------------------------------------------------------------------
# Correção da velocidade radial (CCF com template)
# ----------------------------------------------------------------------------
def ccf(wave, flux, w_min=5300, w_max=5600, template="sun_harps_ganymede.fits",
        rmin=-80, rmax=80, margem=3.0, rv_manual=None):
    '''Devolve (wave_em_repouso, rv_km_s). Com rv_manual não usa o template.'''
    wave, flux = np.ravel(wave), np.ravel(flux)

    if rv_manual is not None:
        rv = float(rv_manual)
    else:
        t_wave, t_flux = hdus(str(HERE / template))        # rebenta se não existir (de propósito)
        m = (wave > w_min) & (wave < w_max) & np.isfinite(flux) & (flux > 0)
        mt = ((t_wave > w_min - margem) & (t_wave < w_max + margem)
              & np.isfinite(t_flux) & (t_flux > 0))
        if m.sum() < 1000 or mt.sum() < 1000:
            raise ValueError(f"Espectro ou template não cobrem {w_min}-{w_max} Å.")

        fs = norm_lin(wave[m], flux[m], w_min, w_max) - 1.0
        fts = norm_lin(t_wave[mt], t_flux[mt], w_min - margem, w_max + margem) - 1.0
        rvs, cc = crosscorrRV(wave[m], fs, t_wave[mt], fts, rmin, rmax, drv=0.5)
        rv = float(rvs[np.argmax(cc)])

        if abs(rv) > 0.9 * rmax:
            warnings.warn(f"RV={rv:.1f} km/s junto ao limite de [{rmin},{rmax}]; "
                          "alarga o intervalo ou verifica o espectro")

    return wave / (1 + rv / C_KMS), rv


# ----------------------------------------------------------------------------
# EW por ajuste gaussiano (lmfit) com erro
# ----------------------------------------------------------------------------
def ew_gauss(wave_l, flux_norm):
    '''Devolve (ew, ew_err, flux_fit) para uma risca de absorção normalizada (contínuo = 1).'''
    flux_inv = 1.0 - flux_norm
    amp = np.max(flux_inv)
    core = wave_l[flux_inv >= amp / 2.0]
    if len(core) > 1:
        mu, stdv = np.mean(core), np.std(core, ddof=1)
    else:
        mu = wave_l[np.argmax(flux_inv)]
        stdv = max(0.1, (wave_l[-1] - wave_l[0]) / 20)

    def gaussian(x, amplitude, mean, sigma):
        return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    const = np.sqrt(2 * np.pi)
    model = Model(gaussian)
    params = model.make_params(amplitude=amp, mean=mu, sigma=stdv)
    params["amplitude"].min = 0
    params["sigma"].min = 1e-10
    result = model.fit(flux_inv, params, x=wave_l)

    A, s = result.params["amplitude"].value, result.params["sigma"].value
    eA, es = result.params["amplitude"].stderr, result.params["sigma"].stderr
    ew = A * s * const

    if eA is None or es is None:
        ew_err = np.nan
    else:
        rho = result.params["amplitude"].correl.get("sigma", 0.0)
        var = const ** 2 * ((s * eA) ** 2 + (A * es) ** 2 + 2.0 * A * s * rho * eA * es)
        ew_err = float(np.sqrt(max(var, 0.0)))

    flux_fit = 1.0 - gaussian(wave_l, A, result.params["mean"].value, s)
    return ew, ew_err, flux_fit


# ----------------------------------------------------------------------------
# Calibração: Teff = polinómio(f(rácio)),  f = R, 1/R ou ln R
# ----------------------------------------------------------------------------
TRANSFORMS = {
    "R":    lambda r: r,
    "invR": lambda r: 1.0 / r,
    "logR": lambda r: np.log(r),
}


def aplicar_calibracao(ratio, linha_cal):
    '''Teff previsto para um rácio medido, dada uma linha de Calibration.csv.'''
    x = TRANSFORMS[linha_cal["transf"]](np.asarray(ratio, float))
    coefs = [linha_cal[f"c{i}"] for i in range(int(linha_cal["grau"]) + 1)]
    return np.polyval(coefs[::-1], x)


def combinar_estimativas(T, sig, nsigma=2.5):
    '''Rejeição iterativa (mediana + MAD) e média pesada 1/sig^2.
    Devolve (teff, disp, keep) ou None se houver menos de 3 estimativas válidas.'''
    T, sig = np.asarray(T, float), np.asarray(sig, float)
    ok = np.isfinite(T) & np.isfinite(sig)
    if ok.sum() < 3:
        return None
    keep = ok.copy()
    for _ in range(20):
        med = np.median(T[keep])
        s = 1.4826 * np.median(np.abs(T[keep] - med))
        if not np.isfinite(s) or s < 1.0:
            s = max(np.median(sig[keep]), 1.0)
        novo = ok & (np.abs(T - med) <= nsigma * s)
        if (novo == keep).all():
            break
        keep = novo

    n = int(keep.sum())
    if n < 3:
        return None

    w = 1 / sig[keep] ** 2
    teff = np.sum(w * T[keep]) / np.sum(w)
    disp = np.sqrt(np.sum(w * (T[keep] - teff) ** 2) / np.sum(w) * n / max(n - 1, 1))
    return float(teff), float(disp), keep


# ----------------------------------------------------------------------------
# Termómetro
# ----------------------------------------------------------------------------
def estimar_teff(fits_path, cal_csv=HERE / "Calibration.csv", cv_csv=HERE / "Calibration_CV.csv",
                 nsigma=2.5, rv_manual=None):
    '''
    Estima a Teff de uma estrela a partir de um espectro (não precisa de estar em repouso).

    1) corrige a velocidade radial (CCF com o template, ou rv_manual);
    2) mede a EW de cada risca da calibração (mesmas janelas) e forma os rácios EW_1/EW_2;
    3) Teff de cada rácio = polinómio(f(rácio)), sem extrapolar fora do intervalo calibrado;
    4) rejeita discrepantes (mediana + MAD) e combina por média pesada (1/scatter_K^2).

    Erro: sigma = RMS da validação cruzada (Calibration_CV.csv), aumentado quando a dispersão
    entre rácios desta estrela excede a típica da calibração.

    Devolve (res, est): res = dict(Teff, sigma, sigma_scatter, rv, n_antes, n_depois);
                        est = DataFrame com o rácio, a Teff de cada rácio e se foi usado.
    '''
    cal = pd.read_csv(cal_csv)
    cv = pd.read_csv(cv_csv).iloc[0]                # se faltar, correr validar_calibracao

    wave, flux = hdus(fits_path)
    wave, rv = ccf(wave, flux, rv_manual=rv_manual)

    janelas = {}
    for _, c in cal.iterrows():
        janelas[round(c["line_1"], 2)] = c["rng_1"]
        janelas[round(c["line_2"], 2)] = c["rng_2"]

    ews = {}
    for line, rng in janelas.items():
        lim_inf, lim_sup = line - rng / 2, line + rng / 2
        m = (wave > lim_inf) & (wave < lim_sup)
        try:
            flux_n = norm_lin(wave[m], flux[m], lim_inf, lim_sup)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ew, ew_err, _ = ew_gauss(wave[m], flux_n)
            fit_warn = [wi for wi in w if issubclass(wi.category, (RuntimeWarning, UserWarning))
                        and "deprecat" not in str(wi.message).lower()]
            bom = (np.isfinite(ew) and np.isfinite(ew_err)
                   and 1e-3 < ew <= 1 and ew_err > 0 and not fit_warn)
            ews[line] = ew if bom else np.nan
        except Exception:
            ews[line] = np.nan

    linhas = []
    for _, c in cal.iterrows():
        ew1, ew2 = ews[round(c["line_1"], 2)], ews[round(c["line_2"], 2)]
        r = ew1 / ew2 if (ew1 > 0 and ew2 > 0) else np.nan
        T = np.nan
        if np.isfinite(r) and c["ratio_min"] <= r <= c["ratio_max"]:
            T = aplicar_calibracao(r, c)
        linhas.append({"ratio": c["ratio"], "valor_racio": r, "Teff": T,
                       "sigma_cal": c["scatter_K"]})
    est = pd.DataFrame(linhas)

    T, sig = est["Teff"].to_numpy(float), est["sigma_cal"].to_numpy(float)
    out = combinar_estimativas(T, sig, nsigma)
    if out is None:
        raise RuntimeError("Menos de 3 rácios com estimativa válida.")
    teff, disp, keep = out
    est["usado"] = keep

    if not (cal["Teff_min"].median() <= teff <= cal["Teff_max"].median()):
        warnings.warn(f"Teff={teff:.0f} K fora do intervalo calibrado "
                      f"[{cal['Teff_min'].median():.0f}, {cal['Teff_max'].median():.0f}] K: "
                      "extrapolação, resultado pouco fiável.")

    sigma = float(cv["rms_K"] * max(1.0, disp / cv["disp_med_K"]))
    return dict(Teff=teff, sigma=sigma, sigma_scatter=disp, rv=float(rv),
                n_antes=int(np.isfinite(T).sum()), n_depois=int(keep.sum())), est


def main():
    ap = argparse.ArgumentParser(description="Teff a partir de rácios de EW.")
    ap.add_argument("fits", help="espectro .fits (Phase 3 ou 1D com WCS)")
    ap.add_argument("--rv", type=float, default=None,
                    help="RV da estrela em km/s (se omitido, usa o template)")
    ap.add_argument("--cal", default=str(HERE / "Calibration.csv"))
    ap.add_argument("--cv", default=str(HERE / "Calibration_CV.csv"))
    ap.add_argument("--detalhe", action="store_true", help="mostra os rácios usados")
    a = ap.parse_args()

    res, est = estimar_teff(a.fits, a.cal, a.cv, rv_manual=a.rv)
    print(f"Teff = {res['Teff']:.0f} ± {res['sigma']:.0f} K   "
          f"(dispersão entre rácios {res['sigma_scatter']:.0f} K | "
          f"{res['n_depois']}/{res['n_antes']} rácios | RV = {res['rv']:.2f} km/s)")
    if a.detalhe:
        print(est[est["usado"]].to_string(index=False))


if __name__ == "__main__":
    main()
