
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ajustar(teff, y, grau, n_sigma=3.0, n_iter=10):
    '''Ajuste polinomial com rejeição iterativa (sigma-clipping). Devolve dict.'''
    ok = np.isfinite(teff) & np.isfinite(y)
    gl = grau + 1
    c0 = np.polyfit(teff[ok], y[ok], grau)
    scat_antes = np.std(y[ok] - np.polyval(c0, teff[ok]), ddof=gl)
 
    m = ok.copy()
    for _ in range(n_iter):
        coef = np.polyfit(teff[m], y[m], grau)
        res = y - np.polyval(coef, teff)
        s = np.std(res[m], ddof=gl)
        novo = ok & (np.abs(res) < n_sigma * s)
        if (novo == m).all():
            break
        m = novo
    coef = np.polyfit(teff[m], y[m], grau)
    scat = np.std(y[m] - np.polyval(coef, teff[m]), ddof=gl)
 
    # derivada numa grelha uniforme em Teff (não depende da ordem das estrelas)
    grelha = np.linspace(teff[m].min(), teff[m].max(), 200)
    d = np.polyval(np.polyder(coef), grelha)
    monotono = bool(np.all(d < 0) or np.all(d > 0))
    slope = float(np.median(d))
    return dict(coef=coef, mask_ok=ok, mask_uso=m, scat_antes=scat_antes, scat=scat,
                slope=slope, monotono=monotono, sigma_T=scat / abs(slope))
 
def avaliar_racios(df, graus=(1, 2, 3), n_sigma=3.0, teff_col="Teff",
                   out_csv="Ratios_Quality.csv"):

    '''
    Evaluates each racio for every of the 3 polynomial degrees selected.
    That is, realizes len(graus) fits each with a diferent polynomial.
    Then returns a csv file: "Ratios_Quality.csv" with all the information about those fits for each ratio
    '''
    teff = df[teff_col].to_numpy(float)
    linhas = []
    for par in [c for c in df.columns if "/" in c]:
        y = df[par].to_numpy(float)
        if (np.isfinite(teff) & np.isfinite(y)).sum() < 20:
            continue
        for g in graus:
            r = ajustar(teff, y, g, n_sigma)
            linhas.append({"ratio": par, "grau": g, "n_total": int(r["mask_ok"].sum()),
                           "n_usados": int(r["mask_uso"].sum()),
                           "scatter_antes": r["scat_antes"], "scatter_depois": r["scat"],
                           "slope_por_K": r["slope"], "monotono": r["monotono"],
                           "sigma_T_K": r["sigma_T"]})
    res = pd.DataFrame(linhas).sort_values(["monotono", "sigma_T_K"],
                                           ascending=[False, True]).reset_index(drop=True)
    res.to_csv(out_csv, index=False)
    return res


def build_calibration(df, res, pairs_csv="Pairs_Ratios.csv", n_sigma=3.0,
                      teff_col="Teff", out_csv="Calibration_Ratios.csv"):

    '''
    This function selects only the best fits obtained in avaliar_racios()
    The selection is based on the racio that offers the lowest std-deviantion of the Teff
    '''
    teff = df[teff_col].to_numpy(float)
    pairs = pd.read_csv(pairs_csv).drop_duplicates("ratio").set_index("ratio")
    # melhor ajuste monótono por rácio (aqui: menor sigma_T; troca pelo critério LOO/BIC)
    best = (res[res["monotono"]].sort_values("sigma_T_K")
               .drop_duplicates("ratio", keep="first"))
    rows = []
    for _, b in best.iterrows():
        par, g = b["ratio"], int(b["grau"])
        r = ajustar(teff, df[par].to_numpy(float), g, n_sigma)
        t = teff[r["mask_uso"]]
        p = pairs.loc[par]
        row = {"ratio": par,
               "line_1": p["line_1"], "sp_1": p["sp_1"], "EP_1": p["EP_1"],
               "line_2": p["line_2"], "sp_2": p["sp_2"], "EP_2": p["EP_2"],
               "fit_type": "linear" if g == 1 else f"poly{g}", "grau": g,
               "Teff_min": t.min(), "Teff_max": t.max(),
               "n_usados": int(r["mask_uso"].sum()),
               "scatter_ratio": r["scat"], "sigma_T_K": r["sigma_T"]}
        for i, c in enumerate(r["coef"][::-1]):      # c0 + c1*T + c2*T^2 + ...
            row[f"c{i}"] = c
        rows.append(row)
    cal = pd.DataFrame(rows)
    cal.to_csv(out_csv, index=False, float_format="%.10e")
    return cal
 
def plot_calibracao(df, cal, top=6, ncols=3, n_sigma=3.0, teff_col="Teff",
                    out="Ratios_final.png"):
    
    '''Painéis dos `top` melhores rácios da calibração final (grau próprio de cada um).'''
    
    cal = cal.sort_values("sigma_T_K").head(top)
    nrows = int(np.ceil(len(cal) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    teff = df[teff_col].to_numpy(float)

    for ax, (_, c) in zip(axes.ravel(), cal.iterrows()):
        y = df[c["ratio"]].to_numpy(float)
        r = ajustar(teff, y, int(c["grau"]), n_sigma)
        fora = r["mask_ok"] & ~r["mask_uso"]
        ax.scatter(teff[r["mask_uso"]], y[r["mask_uso"]], s=12, alpha=.6, label="usados")
        ax.scatter(teff[fora], y[fora], s=22, c="r", marker="x",
                   label=f"rejeitados ({fora.sum()})")
        t = np.linspace(c["Teff_min"], c["Teff_max"], 200)
        ax.plot(t, np.polyval(r["coef"], t), "k-", label=c["fit_type"])
        ax.set_title(f"{c['ratio']}  ({c['sp_1']}/{c['sp_2']})\n"
                     f"σ_T≈{c['sigma_T_K']:.0f} K", fontsize=10)
        ax.set_xlabel("Teff (K)")
        ax.set_ylabel("EW$_1$/EW$_2$")
        ax.legend(fontsize=8)

    for ax in axes.ravel()[len(cal):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
