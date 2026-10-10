import numpy as np
from espectro import hdus
from EW import norm_lin,ew_gauss
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import warnings

def files(folder_path, stars_csv):
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

def auto_measure(df_found, lines, out_csv="Stars_EW.csv"):
    '''
    df_found: DataFrame com as colunas Stars, Teff, path
    lines: dict {comprimento de onda: largura da janela (rng)}
    Devolve (df com uma coluna por risca, dict com os casos suspeitos)
    '''
    ews = []
    strange = {}

    for star, path in zip(df_found["Stars"], df_found["path"]):
        wave, flux = hdus(path)
        ew_star = []

        for line, rng in lines.items():
            lim_inf = line - rng/2
            lim_sup = line + rng/2

            mask = (wave > lim_inf) & (wave < lim_sup)
            wave_m = wave[mask]
            flux_m = flux[mask]

            try:
                flux_n = norm_lin(wave_m, flux_m, lim_inf, lim_sup)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    ew = ew_gauss(wave_m, flux_n)[0]

                if (ew <= 0) or (ew > 1) or w:              # EW fora do intervalo plausível (Å)
                    strange[(star, line)] = f"Poor Gaussian fit: {ew}"

            except Exception as error:
                ew = np.nan
                strange[(star, line)] = f"erro: {error}"

            ew_star.append(ew)
        ews.append(ew_star)

    cols = [f"{l:.2f}" for l in lines]
    df_ew = pd.DataFrame(ews, columns=cols, index=df_found.index)
    df_out = pd.concat([df_found, df_ew], axis=1)

    df_out.to_csv(out_csv, index=False) 
    
    return df_out, strange

def sensitivity(df_found, ew_max=1.0, ep=None):
    '''
    This functions quantifies how the lines choosen depend on temperature
    
    Arguments:
    - df_found: Data Frame that was obtained in the past functions
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
    return table.sort_values("abs_rho", ascending=False).reset_index(drop=True)
