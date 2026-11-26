#%%
import batman
import numpy as np
import matplotlib.pyplot as plt
from lmfit_to_batman import lmfit_to_batman
import lmfit
import pandas as pd
from data_visualization import phase_fold
from astropy.constants import G as G_univ
from astropy import units as u


# Valores do artigo (Tabela 2)
T0       = 55528.86774      # dias, HJD - 2400000
P        = 0.813475         # dias
a_AU     = 0.0142           # UA
Rs_Rsun  = 0.598            # R_estrela em R_sol
AU_Rsun  = 215.032          # UA em R_sol

a_Rs = a_AU * AU_Rsun / Rs_Rsun   # ~5.11, em raios estelares
rp   = np.sqrt(0.0255)            # Raio Planeta in stellar radii

#
# TESTE BATMAN
#

params = batman.TransitParams()
params.t0 = T0
params.per = P      #orbital period
params.rp = rp      #planet radius (in units of stellar radii)
params.a = a_Rs     #semi-major axis (in units of stellar radii)
params.inc = 82.6   #orbital inclination (in degrees)
params.ecc = 0.     #eccentricity
params.w = 90.      #longitude of periastron (in degrees)
params.limb_dark = "quadratic"    # TESS, valores do documento Targets
params.u = [0.5593, 0.0908]     #Limb Darkning


# --- Modelo em função do tempo ---
t = T0 + np.linspace(-0.05, 0.05, 1000)      # tempos à volta do trânsito
m = batman.TransitModel(params, t)
flux = m.light_curve(params)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(t - T0, flux, 'r-', label='modelo')
# dados (descomentar quando os tiveres, no mesmo sistema de tempo que T0):
# ax[0].plot(t_obs - T0, flux_obs, 'k.', ms=3, label='TESS')
ax[0].set_xlabel('t − T0 (d)')
ax[0].set_ylabel('Fluxo normalizado')
ax[0].legend()


# --- Modelo em função da fase ---
tf = T0 + np.linspace(-0.5, 0.5, 5000) * P   # um período completo
mf = batman.TransitModel(params, tf)
ff = mf.light_curve(params)
fase = (tf - T0) / P                          # de -0.5 a 0.5, trânsito em 0

ax[1].plot(fase, ff, 'r-')
# ax[1].plot(fase_obs, flux_obs, 'k.', ms=3)
ax[1].set_xlim(-0.06, 0.06)                   # zoom no trânsito
ax[1].set_xlabel('Fase orbital')

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------------------------------


#Comparação Modelos em phase
#----------------------------

data_obs = pd.read_csv("lc_clean.csv")
t_obs = data_obs["TIME"].to_numpy(float)
ph    = data_obs["FLUX_PH"].to_numpy(float)
flux_cln = data_obs["FLUX"].to_numpy(float)
err_flux_cln = data_obs["ERR_FLUX"].to_numpy(float)
epoca= data_obs["EPOCH"].to_numpy(float)


D = 0.0483                                    # T14 (d)
E_all = np.round((t_obs - T0) / P).astype(int)   # época de cada ponto

rows = []
for E in np.unique(E_all):
    Tt  = T0 + E * P
    sel = (E_all == E) & (np.abs(t_obs - Tt) < 1.5 * D)
    tt, ff, ee = t_obs[sel], flux_cln[sel], err_flux_cln[sel]

    oot = np.abs(tt - Tt) > 0.6 * D            # fora do trânsito, com margem
    # exige pontos fora do trânsito dos dois lados e cobertura do trânsito
    if (tt[oot] < Tt).sum() < 5 or (tt[oot] > Tt).sum() < 5:
        continue
    if (np.abs(tt - Tt) < 0.5 * D).sum() < 0.8 * (D / (2/1440)):   # ajusta à cadência
        continue

    coef = np.polyfit(tt[oot] - Tt, ff[oot], 1)
    base = np.polyval(coef, tt - Tt)
    rows.append(pd.DataFrame({"TIME": tt, "EPOCH": E,
                              "FLUX_NORM": ff / base,
                              "ERR_NORM": ee / base}))

lc = pd.concat(rows, ignore_index=True)
lc["PHASE"] = ((lc["TIME"] - T0) / P + 0.5) % 1 - 0.5
lc.to_csv("lc_transitos_norm.csv", index=False)

# verificação: a mediana fora do trânsito tem de ser ~1
oot_all = np.abs(lc["PHASE"] * P) > 0.6 * D
print("mediana OOT:", np.median(lc.loc[oot_all, "FLUX_NORM"]))


# modelo nos tempos observados (usa o objeto batman, p.ex. o do teste)
model_obs = batman.TransitModel(params, t_obs).light_curve(params)

idx = np.argsort(ph)
plt.plot(ph, flux_cln, '.', color='lightgray', ms=2, label="Dados")
plt.plot(ph[idx], model_obs[idx], 'r-',label="Modelo")
plt.xlim(-0.1, 0.1)
plt.xlabel('Fase orbital'); plt.ylabel('Fluxo normalizado')
plt.title("Modelo batman vs Phase Fold ")
plt.legend()
plt.show()

# épocas com dados e quantos pontos cada uma tem dentro do trânsito (T14 = 0.0483 d)
epocas = np.unique(epoca.astype(int))
n_in = [np.sum(np.abs(t_obs - (T0 + E*P)) < 0.0483/2) for E in epocas]

E_esc = epocas[np.argmax(n_in)]               
Tt    = T0 + E_esc * P                        

sel = np.abs(t_obs - Tt) < 0.1                # só os pontos deste trânsito
plt.plot(t_obs[sel] - Tt, flux_cln[sel], 'k.', label='Dados')
plt.plot(t_obs[sel] - Tt, model_obs[sel], 'r-', label='Modelo')
plt.xlabel('t − Tt (d)'); plt.ylabel('Fluxo normalizado')
plt.legend()
plt.show()



#-------------------
# LMFIT - Ajuste Real.
#--------------------

lc    = pd.read_csv("lc_transitos_norm.csv")
t_obs = lc["TIME"].to_numpy(float)
ph    = lc["PHASE"].to_numpy(float)
flux  = lc["FLUX_NORM"].to_numpy(float)
err   = lc["ERR_NORM"].to_numpy(float)

def residuos(params, t, flux, f_err):
    parbat = lmfit_to_batman(params)
    modelo = batman.TransitModel(parbat, t).light_curve(parbat)
    return (flux - params['f0'].value * modelo) / f_err

params_lm = lmfit.Parameters()
params_lm.add('t0',  value=T0, min=T0-0.025 , max=T0+0.025 ) #Faz variação relativa
params_lm.add('per', value=P , vary=False)
params_lm.add('rp',  value=rp,   min=0, max=1)
params_lm.add('a',   value=a_Rs, min=1)
params_lm.add('inc', value=82.6, max=90)
params_lm.add('ecc', value=0., vary=False)  #Fixo
params_lm.add('w',   value=90., vary=False) #Fixo
params_lm.add('u1',  value=0.5593, vary=False)
params_lm.add('u2',  value=0.0908, vary=False)
params_lm.add('f0',  value=1.0, min=0.9, max=1.01)   # teste de consistência da normalização

result = lmfit.minimize(residuos, params_lm, args=(t_obs, flux, err))
print(lmfit.fit_report(result))

parbat_fit = lmfit_to_batman(result.params)
modelo_fit = result.params['f0'].value * batman.TransitModel(parbat_fit, t_obs).light_curve(parbat_fit)
res = flux - modelo_fit

idx = np.argsort(ph)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(ph, flux, '.', color='lightgray', ms=2, label='Dados')
ax1.plot(ph[idx], modelo_fit[idx], 'r-', label='Modelo ajustado')
ax1.set_ylabel('Fluxo normalizado')
ax1.legend()

ax2.plot(ph, res, '.', color='k', ms=2)
ax2.axhline(0, color='r')
ax2.set_xlabel('Fase orbital')
ax2.set_ylabel('Resíduos')
ax2.set_xlim(-0.1, 0.1)

plt.tight_layout()
plt.show()

print("N trânsitos usados:", lc["EPOCH"].nunique(), "| N pontos:", len(lc))
print("rms resíduos:", np.std(res), "| mediana erro:", np.median(err))



raio_est = Rs_Rsun * u.R_sun
raio_est_jupiter = raio_est.to(u.R_jup)
print("raio_plane",0.15913521*raio_est_jupiter)

#Debug
# T_t = pd.read_csv("Periods.csv")["T_t"]
# print(T_t-59256.21019)

# print(np.std(T_t,ddof=1))


# plt.scatter(t_obs-59256.21019, flux)
# plt.xlim(-0.1,0.1)
# plt.axvline(0,color="red",ls="--")
# plt.show()
# # star_density=3*np.pi
# %%
#
