#%%
import batman
import numpy as np
import matplotlib.pyplot as plt
from lmfit_to_batman import lmfit_to_batman
import lmfit
import pandas as pd
from data_visualization import phase_fold


# Valores do artigo (Tabela 2)
T0       = 55528.86774      # dias, HJD - 2400000
P        = 0.813475         # dias
a_AU     = 0.0142           # UA
Rs_Rsun  = 0.598            # R_sol
AU_Rsun  = 215.032          # R_sol por UA

a_Rs = a_AU * AU_Rsun / Rs_Rsun   # ~5.11, em raios estelares
rp   = np.sqrt(0.0255)            # Raio Planeta em 

#
# TESTE BATMAN
#

params = batman.TransitParams()
params.t0 = T0
params.per = P   #orbital period
params.rp = rp  #planet radius (in units of stellar radii)
params.a = a_Rs
params.inc = 82.6
params.ecc = 0.
params.w = 90.
params.limb_dark = "quadratic"    # TESS, valores do documento Targets
params.u = [0.5593, 0.0908]


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


#Comparação Modelos em phase
#----------------------------

data_obs = pd.read_csv("lc_clean.csv")
t_obs = data_obs["TIME"].to_numpy(float)
ph    = data_obs["FLUX_PH"].to_numpy(float)
flux_cln = data_obs["FLUX"].to_numpy(float)
err_flux_cln = data_obs["ERR_FLUX"].to_numpy(float)
epoca= data_obs["EPOCH"].to_numpy(float)



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

E_esc = epocas[np.argmax(n_in)]               # ou escolhe à mão uma época da lista
Tt    = T0 + E_esc * P                        # agora é um escalar

sel = np.abs(t_obs - Tt) < 0.1                # só os pontos deste trânsito
plt.plot(t_obs[sel] - Tt, flux_cln[sel], 'k.', label='Dados')
plt.plot(t_obs[sel] - Tt, model_obs[sel], 'r-', label='Modelo')
plt.xlabel('t − Tt (d)'); plt.ylabel('Fluxo normalizado')
plt.legend()
plt.show()



#-------------------
# LMFIT - Ajuste Real.
#--------------------

def residuos(params, t, flux, f_err):
    # 1) 'params' = valores de teste do lmfit nesta iteração
    parbat = lmfit_to_batman(params)

    # 2) modelo calculado nos tempos observados
    modelo = batman.TransitModel(parbat, t).light_curve(parbat)

    # 3) um resíduo por ponto
    return (flux - modelo) / f_err





params_lm = lmfit.Parameters()
params_lm.add('t0',  value=T0, vary=False) #Fixar o T0
params_lm.add('per', value=P, vary=False) #Fixar o Periodo
params_lm.add('rp',  value=rp,   min=0, max=1)
params_lm.add('a',   value=a_Rs, min=1, vary=False) #Fixar o semi-eixo maior
params_lm.add('inc', value=82.6, max=90, vary=False) #Fixar a inclinação
params_lm.add('ecc', value=0. , vary=False)
params_lm.add('w',   value=90. , vary=False )
params_lm.add('u1',  value=0.5593)
params_lm.add('u2',  value=0.0908)

result = lmfit.minimize(residuos, params_lm, args=(t_obs, flux_cln, err_flux_cln))
print(lmfit.fit_report(result))

parbat_fit = lmfit_to_batman(result.params)

# modelo nos tempos observados
modelo_fit = batman.TransitModel(parbat_fit, t_obs).light_curve(parbat_fit)

res = flux_cln - modelo_fit        # (13606,) - (13606,)

idx = np.argsort(ph)
plt.plot(ph, flux_cln, '.', color='lightgray', ms=2, label='Dados')
plt.plot(ph[idx], modelo_fit[idx], 'r-', label='Modelo ajustado')
plt.xlim(-0.1, 0.1)
plt.xlabel('Fase orbital'); plt.ylabel('Fluxo normalizado')
plt.legend()
plt.show()
# %%
#
