
import numpy as np
import lmfit
import pandas as pd
import matplotlib.pyplot as plt

def in_phase(t,T0=55528.86774,P=0.813475):
    E=(t-T0)/P

    phase =E%1.0

    phase=np.where(phase >0.5, phase-1, phase)
    return phase,E


T0 = 55528.86774
P = 0.813475

def model_rv(t, gamma, K, T0, P):
    return gamma - K*np.sin(2*np.pi*(t - T0)/P)

df_rv = pd.read_csv("rv_clean.csv")

t_rv = df_rv["rjd"].values     
rv   = df_rv["vrad"].values
err_rv  = df_rv["svrad"].values  
phase=df_rv["RV_PH"] 

gamma0 = np.mean(rv)
K0 = (np.max(rv) - np.min(rv))/2

#FIT - lmfit
model = lmfit.Model(model_rv)
params = model.make_params(gamma=gamma0, K=K0, T0=T0, P=P)
params["T0"].vary = False
params["P"].vary = False

result = model.fit(rv, params, t=t_rv, weights=1/err_rv)
#print(result.fit_report())

gamma=result.params["gamma"].value
K=result.params["K"].value

data_fit=pd.DataFrame({"gamma": gamma, "K": K, "T0": T0, "P": P},index=[0])
data_fit.to_csv("rv_fit_params.csv", index=False)



#Plotting
t=np.linspace(np.min(t_rv),np.max(t_rv), 1000) #new test t_rv
rv_fit=model_rv(t,gamma,K,T0,P) #Calculated radial velocities


# In time domain
# plt.plot(t,rv_fit,ls="--")
# plt.scatter(t_rv,rv,color="orange")
# plt.show()

#In phase domain
new_phase=in_phase(t)[0]
idx = np.argsort(new_phase) #Sorts the phase for plotting


plt.plot(new_phase[idx],rv_fit[idx],label="Fit",color="blue") #Fit curve
plt.errorbar(phase,rv,err_rv,color="orange",label="Data",fmt="o") #Data points
plt.title("Radial Velocity Curve")
plt.xlabel("Phase")
plt.ylabel("Radial Velocity [m/s]")
plt.legend()
plt.savefig(f"Model_RV.png", dpi=200, bbox_inches="tight")

plt.show()


