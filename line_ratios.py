import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt


def load_ep(path="ep_sousa.txt"):
    '''Lê a lista de Sousa: devolve DataFrame com line, EP, loggf, species.'''
    rows = []
    for l in open(path):
        p = l.split()
        if len(p) < 4:                       # linhas em branco
            continue
        wave, ep, loggf, sp = p[-4:]         # funciona com ou sem índice
        try:
            rows.append((float(wave), float(ep), float(loggf), sp))
        except ValueError:                   # cabeçalho ou linha mal formada
            continue
    return pd.DataFrame(rows, columns=["line", "EP", "loggf", "species"])


def make_pairs(ranking_csv="Ranking_Lines_Calib.csv", sousa_file="ep_sousa.txt",
               dlam_max=70, dep_min=3, tol=0.1, out_csv="Pairs_Ratios.csv"):
    '''
    Constrói os pares de riscas candidatos a rácio (PL4, 2a):
      |Δλ| < dlam_max (Å)  e  |ΔEP| > dep_min (eV)

    ranking_csv: output da função sensitivity
    sousa_file : ficheiro com λ, EP, log gf e espécie
    tol        : tolerância (Å) para associar a risca do ranking à de Sousa
    Devolve o DataFrame com os pares; a risca de EP mais baixo é sempre a "lo",
    e o rácio é EW_lo / EW_hi.
    '''
    sousa = load_ep(sousa_file)
    rank = pd.read_csv(ranking_csv).drop(columns="EP", errors="ignore")

    # associa cada risca à de Sousa mais próxima (tolera 5727.00 vs 5727.05)
    rank = pd.merge_asof(rank.sort_values("line"), sousa.sort_values("line"),
                         on="line", direction="nearest", tolerance=tol)

    sem_ep = rank[rank["EP"].isna()]["line"].tolist()
    if sem_ep:
        print("Riscas sem correspondência em Sousa (ignoradas):", sem_ep)
    rank = rank.dropna(subset=["EP"]).reset_index(drop=True)

    pairs = []
    for a, b in combinations(rank.index, 2):
        ra, rb = rank.loc[a], rank.loc[b]
        dlam = abs(ra["line"] - rb["line"])
        dep = abs(ra["EP"] - rb["EP"])

        #Line_1 : lowest EP ; Line_2 : Highest EP
        if dlam < dlam_max and dep > dep_min:
            lo, hi = (ra, rb) if ra["EP"] < rb["EP"] else (rb, ra)
            pairs.append({"ratio": f"{lo['line']:.2f}/{hi['line']:.2f}",
                          "line_1": lo["line"], "sp_1": lo["species"], "EP_1": lo["EP"],
                          "line_2": hi["line"], "sp_2": hi["species"], "EP_2": hi["EP"],
                          "dlam": dlam, "dEP": dep,
                          "rho_1": lo["rho"], "rho_2": hi["rho"]})

    df = pd.DataFrame(pairs)
    df["drho"] = (df["rho_1"] - df["rho_2"]).abs()   # pré-ordenação (heurística)
    df = df.sort_values("drho", ascending=False).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return df

def ratios(Stars_EW,pair, out_csv="Line_Ratios.csv"):
    '''
    Stars_EW : CSV file from auto_measure function (Stars_EW.csv)
    pairs : CSV file from make_pairs function

    Returns:
    - CSV file IDENTICAL to Stars_WE.csv (Swaps the lines for the ratios)

    Note: If the Stars_EW.csv is the output from auto_measure.py, then this output file will already contain the Teff's 
    for each star.
    '''

    ews = pd.read_csv(Stars_EW)
    pairs = pd.read_csv(pair)

    # Maps the wavelength to the columns name of the new CSV; 
    cols = {}
    for c in ews.columns:
        try:
            cols[round(float(c), 2)] = c
        #In case of being names
        except ValueError:
            pass
    meta = [c for c in ews.columns if c not in cols.values()] #Columns that are names: to be used latter

    # Computation of the racios
    novos, em_falta = {}, []

    for _, p in pairs.iterrows():
        c1, c2 = cols.get(round(p["line_1"], 2)), cols.get(round(p["line_2"], 2))
        
        #If anything is missing 
        if c1 is None or c2 is None:
            em_falta.append(p["ratio"])
            continue

        ew1, ew2 = ews[c1], ews[c2]
        valido = (ew1 > 0.0) & (ew2 > 0.0) #Condition to remove NaN's and poor sneaky fits
        novos[p["ratio"]] = (ew1 / ew2).where(valido)

    if em_falta:
        print("Pares sem EW medida (ignorados):", em_falta)

    #Creates new df with old ASCII columns and adds the ratio columns
    df = pd.concat([ews[meta], pd.DataFrame(novos)], axis=1)
    df.to_csv(out_csv, index=False)
    return df



