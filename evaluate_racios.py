import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def loo_rms(x, y, grau):
    '''RMS leave-one-out (em K) de um ajuste polinomial, via matriz chapéu (sem refazer N ajustes).'''
    X = np.vander(x, grau + 1)
    H = X @ np.linalg.pinv(X)
    h = np.diag(H)
    res = y - H @ y
    return float(np.sqrt(np.mean((res / (1 - h)) ** 2)))


def ajustar(ratio, teff, grau, n_sigma=3.0, n_iter=10):
    '''
    Ajuste  Teff = polinómio(rácio)  com rejeição iterativa (sigma-clipping).
    Os resíduos estão em K, logo o scatter é directamente o erro em Teff.
    Devolve dict (coef em ordem np.polyfit: do grau mais alto para o termo constante).
    '''
    ok = np.isfinite(ratio) & np.isfinite(teff)
    gl = grau + 1
    c0 = np.polyfit(ratio[ok], teff[ok], grau)
    scat_antes = np.std(teff[ok] - np.polyval(c0, ratio[ok]), ddof=gl)

    m = ok.copy()
    for _ in range(n_iter):
        coef = np.polyfit(ratio[m], teff[m], grau)
        res = teff - np.polyval(coef, ratio)
        s = np.std(res[m], ddof=gl)
        novo = ok & (np.abs(res) < n_sigma * s)
        if (novo == m).all() or novo.sum() <= gl + 1:
            break
        m = novo
    coef = np.polyfit(ratio[m], teff[m], grau)
    scat = np.std(teff[m] - np.polyval(coef, ratio[m]), ddof=gl)

    # monotonia de Teff(rácio) ao longo do intervalo de rácios usado
    grelha = np.linspace(ratio[m].min(), ratio[m].max(), 200)
    d = np.polyval(np.polyder(coef), grelha)
    monotono = bool(np.all(d < 0) or np.all(d > 0))
    return dict(coef=coef, mask_ok=ok, mask_uso=m, scat_antes=scat_antes, scat=scat,
                slope=float(np.median(d)),            # K por unidade de rácio
                monotono=monotono, sigma_T=scat)


def avaliar_racios(df, graus=(1, 2, 3), n_sigma=3.0, teff_col="Teff",
                   out_csv="Ratios_Quality.csv"):
    teff = df[teff_col].to_numpy(float)
    linhas = []
    for par in [c for c in df.columns if "/" in c]:
        y = df[par].to_numpy(float)
        if (np.isfinite(teff) & np.isfinite(y)).sum() < 20:
            continue
        for g in graus:
            r = ajustar(y, teff, g, n_sigma)
            m = r["mask_uso"]
            linhas.append({"ratio": par, "grau": g,
                           "n_total": int(r["mask_ok"].sum()), "n_usados": int(m.sum()),
                           "scatter_antes_K": r["scat_antes"], "scatter_depois_K": r["scat"],
                           "loo_K": loo_rms(y[m], teff[m], g),
                           "slope_K_por_unid": r["slope"], "monotono": r["monotono"],
                           "sigma_T_K": r["sigma_T"]})
    res = (pd.DataFrame(linhas)
             .sort_values(["monotono", "loo_K"], ascending=[False, True])
             .reset_index(drop=True))
    res.to_csv(out_csv, index=False)
    return res


def build_calibration(df, res, lines, pairs_csv="Pairs_Ratios.csv", n_sigma=3.0,
                      teff_col="Teff", out_csv="Calibration.csv"):
    '''
    Ficheiro de calibração (1 linha por rácio):  Teff = c0 + c1*r + c2*r^2 + ...
    lines: o dicionário {λ: rng} usado no auto_measure -> guarda-se a janela de cada risca
           para o termómetro medir as EW exactamente como na calibração.
    Para cada rácio escolhe-se o grau monótono com menor erro leave-one-out.
    '''
    teff = df[teff_col].to_numpy(float)
    pairs = pd.read_csv(pairs_csv).drop_duplicates("ratio").set_index("ratio")
    rng_map = {round(float(k), 2): v for k, v in lines.items()}
    best = (res[res["monotono"]].sort_values("loo_K")
               .drop_duplicates("ratio", keep="first"))
    rows = []
    for _, b in best.iterrows():
        par, g = b["ratio"], int(b["grau"])
        y = df[par].to_numpy(float)
        r = ajustar(y, teff, g, n_sigma)
        m = r["mask_uso"]
        p = pairs.loc[par]
        row = {"ratio": par,
               "line_1": p["line_1"], "sp_1": p["sp_1"], "EP_1": p["EP_1"],
               "rng_1": rng_map[round(p["line_1"], 2)],
               "line_2": p["line_2"], "sp_2": p["sp_2"], "EP_2": p["EP_2"],
               "rng_2": rng_map[round(p["line_2"], 2)],
               "fit_type": "linear" if g == 1 else f"poly{g}", "grau": g,
               "ratio_min": y[m].min(), "ratio_max": y[m].max(),
               "Teff_min": teff[m].min(), "Teff_max": teff[m].max(),
               "n_usados": int(m.sum()),
               "scatter_K": r["scat"], "loo_K": b["loo_K"]}
        for i, c in enumerate(r["coef"][::-1]):          # c0 + c1*r + c2*r^2 + ...
            row[f"c{i}"] = c
        rows.append(row)
    cal = pd.DataFrame(rows)
    cal.to_csv(out_csv, index=False, float_format="%.10g")
    return cal


def plot_calibracao(df, cal, top=6, ncols=3, n_sigma=3.0, teff_col="Teff",
                    out="Ratios_final.png"):
    '''Calibração clássica Teff vs rácio para os `top` melhores rácios (grau próprio de cada um).'''
    cal = cal.sort_values("loo_K").head(top)
    nrows = int(np.ceil(len(cal) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    teff = df[teff_col].to_numpy(float)

    for ax, (_, c) in zip(axes.ravel(), cal.iterrows()):
        y = df[c["ratio"]].to_numpy(float)
        r = ajustar(y, teff, int(c["grau"]), n_sigma)
        fora = r["mask_ok"] & ~r["mask_uso"]
        ax.scatter(y[r["mask_uso"]], teff[r["mask_uso"]], s=12, alpha=.6, label="usados")
        ax.scatter(y[fora], teff[fora], s=22, c="r", marker="x",
                   label=f"rejeitados ({fora.sum()})")
        x = np.linspace(c["ratio_min"], c["ratio_max"], 200)
        ax.plot(x, np.polyval(r["coef"], x), "k-", label=c["fit_type"])
        ax.set_title(f"{c['ratio']}  ({c['sp_1']}/{c['sp_2']})\n"
                     f"σ≈{c['scatter_K']:.0f} K", fontsize=10)
        ax.set_xlabel("EW$_1$/EW$_2$")
        ax.set_ylabel("Teff (K)")
        ax.legend(fontsize=8)

    for ax in axes.ravel()[len(cal):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)





