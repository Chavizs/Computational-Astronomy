
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
 
def plot_racios(df, res, grau=2, n_sigma=3.0, top=12, teff_col="Teff", ncols=4):
    '''Painéis só para os `top` melhores rácios (ao grau escolhido).'''
    pares = res[res["grau"] == grau]["ratio"].head(top).tolist()
    nrows = int(np.ceil(len(pares) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
    teff = df[teff_col].to_numpy(float)
    for ax, par in zip(axes.ravel(), pares):
        y = df[par].to_numpy(float)
        r = ajustar(teff, y, grau, n_sigma)
        fora = r["mask_ok"] & ~r["mask_uso"]
        ax.scatter(teff[r["mask_uso"]], y[r["mask_uso"]], s=8, alpha=.5)
        ax.scatter(teff[fora], y[fora], s=14, c="r", label="outliers")
        t = np.linspace(teff[r["mask_ok"]].min(), teff[r["mask_ok"]].max(), 200)
        ax.plot(t, np.polyval(r["coef"], t), "k-")
        ax.set_title(f"{par}  σ_T≈{r['sigma_T']:.0f} K", fontsize=9)
        ax.set_xlabel("Teff (K)")
    for ax in axes.ravel()[len(pares):]:
        ax.set_visible(False)
    fig.tight_layout(); fig.savefig("Ratios_vs_Teff.png", dpi=150)
