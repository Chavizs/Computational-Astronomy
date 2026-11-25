#%%
import numpy as np
import lmfit
import pandas as pd
import matplotlib.pyplot as plt
from astropy.constants import G as G_univ
from scipy.stats import chisquare
from uncertainties import ufloat
from uncertainties.umath import *  # sin(), etc.


# TODO fazer o graf dos resíduos num painel abaxio do ajuste 
def in_phase(t,T0=55528.86774,P=0.813475):
    E=(t-T0)/P

    phase =E%1.0

    phase=np.where(phase >0.5, phase-1, phase)
    return phase,E


T0 = 55528.86774#s
P =  0.813475#day
i = np.deg2rad(82.6) # inclination angle
M_star=0.58 #Solar masses

def model_rv(t, gamma, K, T0, P):
    return gamma - K*np.sin(2*np.pi*(t - T0)/P)

#Data from Coralie
df_rv = pd.read_csv("rv_clean.csv")

t_rv = df_rv["rjd"].values     
rv   = df_rv["vrad"].values
err_rv  = df_rv["svrad"].values  
phase=df_rv["RV_PH"] 

gamma0 = np.mean(rv)
K0 = (np.max(rv) - np.min(rv))/2

#-----------
#FIT - lmfit
#-------------
model = lmfit.Model(model_rv)
params = model.make_params(gamma=gamma0, K=K0, T0=T0, P=P)
params["T0"].vary = False
params["P"].vary = False

result = model.fit(rv, params, t=t_rv, weights=1/err_rv)
print(result.fit_report())
#Parameters obtained and their respective error associated
gamma, err_gamma =result.params["gamma"].value, result.params["gamma"].stderr #em m/s
K, err_K=result.params["K"].value, result.params["K"].stderr

chi2 = result.chisqr
red_chi2 = result.redchi



#Mp determination TODO
t0 = ufloat(55528.86774,0.00014) #s
p = ufloat(0.813475,0.000001) #day
k= ufloat(K,err_K)
m_star=ufloat(M_star,0.05)*1.98847e30

Mp= k*np.pow(m_star,2/3) * np.pow( p*24*3600/ (2*np.pi*G_univ), 1/3 )/np.sin(i)#In Jupyter masses
Mp=(Mp/1.89813e27).value 
print(Mp)



#-----------
#Plotting
#----------
t=np.linspace(np.min(t_rv),np.max(t_rv), 1000) #new test t_rv
rv_fit=model_rv(t,gamma,K,T0,P) #Calculated radial velocities


# In time domain
# plt.plot(t,rv_fit,ls="--")
# plt.scatter(t_rv,rv,color="orange")
# plt.show()

#In phase domain
new_phase=in_phase(t)[0]
idx = np.argsort(new_phase) #Sorts the phase for plotting

#To Plot with the y axis starting in 0 do: -gamma to the rv's
plt.plot(new_phase[idx],rv_fit[idx]-gamma,label="Fit",color="blue") #Fit curve
plt.errorbar(phase,rv-gamma,err_rv,color="orange",label="Data",fmt="o") #Data points
plt.title("Radial Velocity Curve")
plt.xlabel("Phase")
plt.ylabel("Radial Velocity [m/s]")
plt.legend()
plt.savefig(f"Model_RV.png", dpi=200, bbox_inches="tight")

plt.show()

#Residues

ress=rv-model_rv(t_rv, gamma,K,T0,P)
std=np.std(ress)

plt.scatter(phase,ress,color="orange")
#plt.ylim(np.min(rv)-gamma,np.max(rv)-gamma)
plt.ylim(-100,100)
# plt.axhline(std,color="red",ls="--")
# plt.axhline(-std,color="red",ls="--")
plt.title("Resíduos do ajuste da velocidade radial")
plt.xlabel("Phase")
plt.ylabel("Resíduos da velocidade radial [m/s]")
plt.legend()
plt.savefig(f"ress_rv.png", dpi=200, bbox_inches="tight")

plt.show()


data_fit=pd.DataFrame({"chi2":chi2, "red_chi2":red_chi2 , "gamma": gamma, "err_gamma":err_gamma, "K": K, "err_K":err_K, "T0": T0, "P": P , "Mp/Mj":Mp},index=[0])
data_fit.to_csv("rv_fit_params.csv", index=False)







#%%

