#%%
import numpy as np
from espectro import hdus
from EW import norm_lin,ew_gauss
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import warnings

def files(folder_path, stars_csv):

    '''
    This file returns the file paths for each star selected
    '''
    folder = Path(folder_path)
    df = pd.read_csv(stars_csv)
    df["Stars"] = df["Stars"].astype(str).str.strip()

    map_fits = {fil.name: fil for fil in folder.rglob("*_rv.fits")}

    paths = df["Stars"].map(lambda s: map_fits.get(f"{s}_rv.fits"))
    found = paths.notna()

    if (~found).any():
        print("Não encontradas:", list(df.loc[~found, "Stars"]))

    df_found = df[found].copy()
    df_found["path"] = [str(p) for p in paths[found]]
    return df_found  # colunas: Stars, Teff, path


def auto_measure(df_found, lines, out_csv="Stars_EW.csv", strange_csv="Strange_fits.csv",
                 err_csv="Stars_EWerr.csv", max_rel_err=None):
    '''
    Mede o equivalente width (EW) de cada linha para cada estrela.

    Parameters
    ----------
    df_found : DataFrame
        Colunas: Stars, Teff, path.
    lines : dict
        {comprimento_de_onda: largura_da_janela (rng)}.
    out_csv : str
        Ficheiro onde guardar a tabela de EWs (poor fits = NaN).
    strange_csv : str
        Ficheiro onde guardar a tabela de casos suspeitos.

    Returns
    -------
    df_out : DataFrame
        df_found + uma coluna por linha (poor fits = NaN).
    strange : dict
        {(star, line): {"rng": float, "error": str}}.
    '''

    # --- aviso de duplicados (Python já só guarda o último) ---
    seen = {}
    for k in lines:
        seen[k] = seen.get(k, 0) + 1
    dups = [k for k, n in seen.items() if n > 1]
    if dups:
        print(f"AVISO: {len(dups)} linhas duplicadas no dicionário 'lines' "
              f"(ficou o último rng): {dups}")

    ews, errs = [], []
    strange = {}

    for star, path in zip(df_found["Stars"], df_found["path"]):
        wave, flux = hdus(path)
        ew_star, err_star = [], [] 

        for line, rng in lines.items():
            lim_inf = line - rng / 2
            lim_sup = line + rng / 2

            mask = (wave > lim_inf) & (wave < lim_sup)
            wave_m = wave[mask]
            flux_m = flux[mask]

            ew = np.nan
            ew_err = np.nan

            # --- normalização ---
            try:
                flux_n = norm_lin(wave_m, flux_m, lim_inf, lim_sup)

            except Exception as error:
                strange[(star, line)] = {
                    "rng": rng,
                    "error": f"norm_lin: {type(error).__name__}: {error}"
                }
                ew_star.append(ew); err_star.append(ew_err)
                continue

            # --- ajuste gaussiano ---
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = ew_gauss(wave_m, flux_n)

                # aceita 2 ou 3 valores
                ew, ew_err = result[0], result[1]
                if not np.isfinite(ew):
                    raise ValueError(f"ew não finito: {ew}")

                # warnings que interessam (fit / runtime), não deprecations
                fit_warn = [
                    wi for wi in w
                    if issubclass(wi.category, (RuntimeWarning, UserWarning))
                    and "deprecat" not in str(wi.message).lower()
                ]

                # --- critério de "poor fit" ---
                ew_mau = (ew <= 1e-3) or (ew > 1) or not np.isfinite(ew)
                err_mau = (not np.isfinite(ew_err)) or (ew_err <= 0) \
                    or ((max_rel_err is not None) and not (ew_err / ew <= max_rel_err))
                    
                if ew_mau or fit_warn or err_mau:   

                    msg = f"Poor Gaussian fit: {ew:.4f}"
                    if err_mau:
                        msg += f" | err/EW = {ew_err / ew:.2f}"
                    if fit_warn:
                        msg += f" | warn: {fit_warn[0].message}"
                    strange[(star, line)] = {"rng": rng, "error": msg}
                    ew = np.nan; ew_err = np.nan               # <-- NÃO entra no df_out

            except Exception as error:
                ew = np.nan; ew_err = np.nan
                strange[(star, line)] = {
                    "rng": rng,
                    "error": f"ew_gauss: {type(error).__name__}: {error}"
                }

            ew_star.append(ew); err_star.append(ew_err) 

        ews.append(ew_star); errs.append(err_star)

    # --- construir DataFrame de saída ---
    cols = [f"{l:.2f}" for l in lines]
    df_ew  = pd.DataFrame(ews,  columns=cols, index=df_found.index)
    df_err = pd.DataFrame(errs, columns=cols, index=df_found.index)

    df_out = pd.concat([df_found, df_ew], axis=1)
    df_out.to_csv(out_csv, index=False)
    pd.concat([df_found, df_err], axis=1).to_csv(err_csv, index=False)

    # --- tabela dos casos suspeitos, ordenada ---
    if strange:

        df_strange = pd.DataFrame(
            [{"star": s, "line": l, "rng": v["rng"], "error": v["error"]}
             for (s, l), v in strange.items()]
        ).sort_values(["star", "line"]).reset_index(drop=True)

        df_strange.to_csv(strange_csv, index=False)
        print(f"{len(strange)} casos suspeitos guardados em {strange_csv}")

    else:
        print("Sem casos suspeitos.")

    return df_out, strange

def sensitivity(df_found, ew_max=1.0, ep=None):
    '''
    This functions quantifies how the lines choosen depend on temperature
    
    Arguments:
    - df_found: Data Frame that was obtained in the past functions: Stars_EW.csv
    - ew_max : Maximum Equivalent Width acceptable (ortherwise will be removed)
    -ep : Excitattion Potencial

    Returns:
    - Table with the lines, coeficients of the statistical analiliys and the ep.
    The lines are in descending order: More Temperature dependent to least dependent
    '''
    #Esctrascts lines (in an array) and the Teff
    lines = np.array(df_found.columns)[3:]
    teff_all = df_found["Teff"].astype(float)
    
    rows = []
    for line in lines:
        ew = df_found[line].astype(float)
        ok = ew.notna() & (ew > 0) & (ew < ew_max)
        x, y = teff_all[ok], ew[ok] * 1000

        if len(x) < 5:
            print(f"{line}: só {len(x)} pontos válidos, ignorada")
            continue

        slope = np.polyfit(x, y, 1)[0]
        rho = x.corr(y, method="spearman")
        change = slope * (x.max() - x.min()) / y.mean()

        rows.append({"line": float(line), "N": len(x), "rho": rho,
                     "slope_mA_per_K": slope, "rel_change": change,
                     "EP": ep.get(float(line)) if ep else np.nan})

    table = pd.DataFrame(rows)
    table["abs_rho"] = table["rho"].abs()
    table = table.sort_values("abs_rho", ascending=False).reset_index(drop=True)

    table.to_csv("Ranking_Lines_Calib.csv", index=False)

    return table

lines={6173.34:3, 6358.68:3, 6380.75:3, 5787.92:3, 
        6481.88:3, 5054.65:2, 
        5965.84:3, 5978.55:3, 5977:3, 5727:3,
        5543.94:1, 5010.94:1, 5688.22:1, 5711.09:1, 5690.43:1, 5772.15:1, 5948.54:1, 6142.49:1, 6145.02:1, 6237.33:1 , 6243.82:1, 6244.48:1, 6527.21:1, 
        5867.56:1, 6166.44:1, 6449.82:1, 6455.6:1, 6471.67:1, 6499.65:1, 5526.82:1, 6245.62:1, 6604.6:1, 4645.19:1, 4820.41:1, 
        5039.96:1, 5866.45:1, 6258.11:1, 5381.03:1, 5418.77:1, 4708.02:1, 5296.7:1, 5578.72:1, 5805.22:1, 
        6086.29:1, 6108.12:1, 6128.98:1, 6175.37:1, 6186.72:1, 6378.26:1, 6635.13:1, 6767.78:1, 6772.32:1, 4602.0:1, 4602.95:1, 
        4620.51:1,  4625.05:1, 4629.34:1, 4630.12:1, 4772.82:1, 4788.76:1,6154.23:1, 6160.75:1, 6318.72:1,  6195.46:1, 6721.85:1, 
        5016.17:1, 4739.11:1, 4626.18:1 , 5342.71:1, 5081.11:1,  5847.0:1, 6111.08:1,  6176.82:1,  6842.04:1, 4508.28:1, 
        4523.4:1, 4561.41:1,  4576.34:1 , 4757.58:1, 
        4802.52:2, 4802.88:2, 4946.39:1, 4962.58:2,  4994.14:2, 5044.22:2, 5067.15:2, 5068.77:2, 5074.76:1, 
        5090.78:2,  5127.36:2,  5159.06:2, 5180.06:2, 5198.72:2,  5229.86:2, 5242.50:2, 5243.78:2,   5253.47:2, 
        5288.53:2, 5373.71:2, 5379.58:2,  5389.48:2, 5417.04:2 , 5441.34:2,  5522.45:2 , 5560.22:2,  5633.95:2, 
        5653.87:2 ,5667.52:1 ,  5679.03:2, 5701.55:2 , 5715.09:1 , 5731.77:2 ,  5741.85:2, 5752.04:2,  5775.08:2,  5778.46:2, 
        5806.73:2, 5809.22:2,  5852.22:2, 5856.09:2 , 5862.36:2, 5905.68:2, 5916.26:2, 5927.79:2, 5934.66:2, 5956.70:2, 5983.69:2, 
        5984.82:2, 5987.07:2, 6005.55:2, 6024.06:2, 6027.06:2, 6056.01:2, 6065.49:2, 6082.72:2, 6096.67:2, 6127.91:2, 6157.73:2 }




if __name__=="__main__":
    df=files("star_fits","estrelas_selecionadas.csv")
    df,strange=auto_measure(df,lines)
    print(strange)
    ranking = sensitivity(df)
    print(ranking)





# %%
