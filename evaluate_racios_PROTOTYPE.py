import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Transformações candidatas do rácio R:  Teff = polinómio( f(R) )
#   f   : transformação aplicada ao rácio
#   df  : derivada df/dR (regra da cadeia -> dTeff/dR, para o slope e a monotonia)
#   dom : máscara dos valores de R onde f(R) está definida
#   tag : prefixo usado em fit_type ("R_poly2", "invR_linear", "logR_poly3", ...)
# ----------------------------------------------------------------------------
TRANSFORMS = {
    "R":    dict(f=lambda r: r,
                 df=lambda r: np.ones_like(r),
                 dom=lambda r: np.isfinite(r)),
    "invR": dict(f=lambda r: 1.0 / r,
                 df=lambda r: -1.0 / r**2,
                 dom=lambda r: np.isfinite(r) & (r != 0)),
    "logR": dict(f=lambda r: np.log(r),
                 df=lambda r: 1.0 / r,
                 dom=lambda r: np.isfinite(r) & (r > 0)),
}


def nome_ajuste(transf, grau):
    return f"{transf}_" + ("linear" if grau == 1 else f"poly{grau}")


def loo_rms(x, y, grau):
    '''RMS leave-one-out (em K) de um ajuste polinomial, via matriz chapéu (sem refazer N ajustes).
    x é a variável JÁ transformada (R, 1/R ou log R).'''
    X = np.vander(x, grau + 1)
    H = X @ np.linalg.pinv(X)
    h = np.diag(H)
    res = y - H @ y
    return float(np.sqrt(np.mean((res / (1 - h)) ** 2)))


def ajustar(ratio, teff, grau, transf="R", n_sigma=3.0, n_iter=10):
    '''
    ratio : Line_Ratios.csv from ratios()

    Ajuste  Teff = polinómio(f(rácio))  com rejeição iterativa (sigma-clipping),
    onde f depende de `transf` ("R", "invR" ou "logR").
    Os resíduos estão em K, logo o scatter é directamente o erro em Teff.
    Devolve dict (coef em ordem np.polyfit: do grau mais alto para o termo constante;
    os coeficientes são em função de x = f(rácio)).
    '''
    T = TRANSFORMS[transf]
    ratio = np.asarray(ratio, float)
    ok = T["dom"](ratio) & np.isfinite(teff)

    x = np.full_like(ratio, np.nan)
    with np.errstate(all="ignore"):
        x[ok] = T["f"](ratio[ok])
    ok &= np.isfinite(x)                      # protege contra inf/nan da transformação
    gl = grau + 1

    c0 = np.polyfit(x[ok], teff[ok], grau)
    scat_antes = np.std(teff[ok] - np.polyval(c0, x[ok]), ddof=gl)

    m = ok.copy()
    for _ in range(n_iter):
        coef = np.polyfit(x[m], teff[m], grau)
        res = teff - np.polyval(coef, x)
        s = np.std(res[m], ddof=gl)
        novo = ok & (np.abs(res) < n_sigma * s)
        if (novo == m).all() or novo.sum() <= gl + 1:
            break
        m = novo
    coef = np.polyfit(x[m], teff[m], grau)
    scat = np.std(teff[m] - np.polyval(coef, x[m]), ddof=gl)

    # monotonia de Teff(rácio): grelha uniforme em R; dTeff/dR = dTeff/dx * dx/dR
    # (f é monótona no domínio, logo monotonia em x <=> monotonia em R)
    grelha = np.linspace(ratio[m].min(), ratio[m].max(), 200)
    d = np.polyval(np.polyder(coef), T["f"](grelha)) * T["df"](grelha)
    monotono = bool(np.all(d < 0) or np.all(d > 0))
    return dict(coef=coef, x=x, transf=transf, mask_ok=ok, mask_uso=m,
                scat_antes=scat_antes, scat=scat,
                slope=float(np.median(d)),            # K por unidade de rácio
                monotono=monotono, sigma_T=scat)


def escolher_ajustes(res, tol_K=2.0):
    '''
    Para cada rácio: o ajuste monótono com menor loo_K, EXCEPTO se o melhor ajuste em R
    ficar a menos de tol_K desse mínimo; nesse caso fica o de R (default, mais simples).
    '''
    mono = res[res["monotono"]]
    escolhidos = []
    for par, g in mono.groupby("ratio"):
        g = g.sort_values("loo_K")
        melhor_global = g.iloc[0]
        gR = g[g["transf"] == "R"]

        if len(gR):
            melhorR = gR.iloc[0]
            ganho = float(melhorR["loo_K"] - melhor_global["loo_K"])
            if ganho <= tol_K:
                esc, motivo = melhorR, "R (default)"
            else:
                esc, motivo = melhor_global, "transf. ganha"
            loo_R = float(melhorR["loo_K"])
        else:
            esc, motivo, loo_R, ganho = melhor_global, "sem R monótono", np.nan, np.nan

        row = esc.to_dict()
        row.update(loo_K_melhor_global=float(melhor_global["loo_K"]),
                   loo_K_R=loo_R, ganho_K=ganho, escolha=motivo)
        escolhidos.append(row)

    if not escolhidos:
        return res.iloc[0:0].assign(
            loo_K_melhor_global=[], loo_K_R=[], ganho_K=[], escolha=[])
    return pd.DataFrame(escolhidos).reset_index(drop=True)


def avaliar_racios(df, graus=(1, 2, 3), transfs=("R", "invR", "logR"), n_sigma=3.0,
                   teff_col="Teff", out_csv="Ratios_Quality_new.csv"):
    '''Testa, para cada rácio, todas as combinações (transformação x grau).'''
    teff = df[teff_col].to_numpy(float)
    linhas = []
    for par in [c for c in df.columns if "/" in c]:
        y = df[par].to_numpy(float)
        if (np.isfinite(teff) & np.isfinite(y)).sum() < 20:
            continue
        for t in transfs:
            for g in graus:
                try:
                    r = ajustar(y, teff, g, t, n_sigma)
                    m = r["mask_uso"]
                    loo = loo_rms(r["x"][m], teff[m], g)
                except (np.linalg.LinAlgError, ValueError):
                    continue                       # ajuste impossível (poucos pontos, etc.)
                linhas.append({"ratio": par, "transf": t, "grau": g,
                               "fit_type": nome_ajuste(t, g),
                               "n_total": int(r["mask_ok"].sum()), "n_usados": int(m.sum()),
                               "scatter_antes_K": r["scat_antes"], "scatter_depois_K": r["scat"],
                               "loo_K": loo,
                               "slope_K_por_unid": r["slope"], "monotono": r["monotono"],
                               "sigma_T_K": r["sigma_T"]})
    res = (pd.DataFrame(linhas)
             .sort_values(["monotono", "loo_K"], ascending=[False, True])
             .reset_index(drop=True))
    res.to_csv(out_csv, index=False)
    return res


def build_calibration(df, res, lines, pairs_csv="Pairs_Ratios.csv", n_sigma=3.0,
                      tol_K=4.0, teff_col="Teff", out_csv="Calibration_new.csv"):
    '''
    df        : DataFrame de "Line_Ratios.csv" (o mesmo usado em avaliar_racios).
    res       : DataFrame "Ratios_Quality.csv" devolvido por avaliar_racios (mesm df, mesmo n_sigma).
    lines     : dict {λ: rng} de auto_measure.py (as janelas ficam gravadas no CSV).
    pairs_csv : "Pairs_Ratios.csv" (de make_pairs; dá espécie e EP de cada risca).

    Ficheiro de calibração (1 linha por rácio):  Teff = c0 + c1*x + c2*x^2 + ...
    com x = f(rácio), onde f é dada pela coluna `transf` ("R" -> x=R, "invR" -> x=1/R, "logR" -> x=ln R).
    lines: o dicionário {λ: rng} usado no auto_measure -> guarda-se a janela de cada risca
           para o termómetro medir as EW exatamente como na calibração.
    Para cada rácio escolhe-se o melhor (transformação, grau) monótono: menor erro leave-one-out.
    '''

    teff = df[teff_col].to_numpy(float)
    pairs = pd.read_csv(pairs_csv).drop_duplicates("ratio").set_index("ratio")
    rng_map = {round(float(k), 2): v for k, v in lines.items()}
    # Primeiro escolhemos o melhor ajuste monótono por rácio (menor loo_K)
    best = escolher_ajustes(res, tol_K=tol_K) 
    rows = []
    for _, b in best.iterrows():
        par, g, t = b["ratio"], int(b["grau"]), b["transf"]
        y = df[par].to_numpy(float)
        r = ajustar(y, teff, g, t, n_sigma)
        m = r["mask_uso"]
        p = pairs.loc[par]
        row = {"ratio": par,
               "line_1": p["line_1"], "sp_1": p["sp_1"], "EP_1": p["EP_1"],
               "rng_1": rng_map[round(p["line_1"], 2)],
               "line_2": p["line_2"], "sp_2": p["sp_2"], "EP_2": p["EP_2"],
               "rng_2": rng_map[round(p["line_2"], 2)],
               "transf": t, "fit_type": nome_ajuste(t, g), "grau": g,
               "ratio_min": y[m].min(), "ratio_max": y[m].max(),   # intervalo em R (não em x)
               "Teff_min": teff[m].min(), "Teff_max": teff[m].max(),
               "n_usados": int(m.sum()),
               "scatter_K": r["scat"], "loo_K": b["loo_K"]}
        for i, c in enumerate(r["coef"][::-1]):          # c0 + c1*x + c2*x^2 + ...
            row[f"c{i}"] = c
        rows.append(row)
    cal = pd.DataFrame(rows)
    cal.to_csv(out_csv, index=False, float_format="%.10g")
    return cal


def aplicar_calibracao(ratio, linha_cal):
    '''
    Teff previsto para um rácio medido, dada uma linha de Calibration.csv.
    (útil para o termómetro: aplica a transformação certa antes do polinómio)

    '''
    x = TRANSFORMS[linha_cal["transf"]]["f"](np.asarray(ratio, float))
    coefs = [linha_cal[f"c{i}"] for i in range(int(linha_cal["grau"]) + 1)]
    return np.polyval(coefs[::-1], x)


def plot_calibracao(df, cal, top=6, ncols=3, n_sigma=3.0, teff_col="Teff",
                    out="Ratios_final_new.png"):
    '''Calibração clássica Teff vs rácio para os `top` melhores rácios (transf. e grau próprios de cada um).

    df : DataFrame de "Line_Ratios.csv".
    cal: DataFrame "Calibration.csv" devolvido por build_calibration

    '''
    cal = cal.sort_values("loo_K").head(top)
    nrows = int(np.ceil(len(cal) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    teff = df[teff_col].to_numpy(float)

    for ax, (_, c) in zip(axes.ravel(), cal.iterrows()):
        y = df[c["ratio"]].to_numpy(float)
        r = ajustar(y, teff, int(c["grau"]), c["transf"], n_sigma)
        fora = r["mask_ok"] & ~r["mask_uso"]
        ax.scatter(y[r["mask_uso"]], teff[r["mask_uso"]], s=12, alpha=.6, label="usados")
        ax.scatter(y[fora], teff[fora], s=22, c="r", marker="x",
                   label=f"rejeitados ({fora.sum()})")
        xr = np.linspace(c["ratio_min"], c["ratio_max"], 200)
        xt = TRANSFORMS[c["transf"]]["f"](xr)
        ax.plot(xr, np.polyval(r["coef"], xt), "k-", label=c["fit_type"])
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
