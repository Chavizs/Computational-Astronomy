
'''
Programa para obter os dados e gráficos de fase e no tempo
'''

import numpy as np
from astropy.io import fits
import sys 
import pandas as pd
import matplotlib.pyplot as plt
from cleanlc import clean_subt_activity

T0=55528.86774
P=0.813475

def info():

    # if (sys.argv[1][-5:] != ".fits" ) or (sys.argv[2][-4:] != ".rbd"):
    #     return "One of the filetypes is not supported or were entered in the wrong order."
    # elif len(sys.argv) >3:
    #     return "Too many files provided."
    # else:
    #     return sys.argv[1],sys.argv[2]
    try:
        fit=sys.argv[1]
        return sys.argv[1],sys.argv[2]
    except:
        return 'tess2021039152502-s0035-0000000036734222-0205-s_lc.fits' , 'WASP-43_CORALIE_1.rdb'

def file_opening(filename):
    
    hdu_list = fits.open(filename)

    hdu_data = hdu_list[1].data
    time = hdu_data["time"]+57000.0
    flux = hdu_data["PDCSAP_FLUX"]
    ferr = hdu_data["PDCSAP_FLUX_ERR"]
    quality = hdu_data["QUALITY"]


    flag = np.where(np.logical_and(  np.equal(quality, 0) , np.greater( flux,  0) ) )[0]
    t = time[flag]
    Flux = flux[flag] #flux
    err_Flux = ferr[flag] #/np.mean(y)

    return t,Flux,err_Flux

def rad_vel(file_rad):

    df_rv=pd.read_csv(file_rad,sep="\t")

    #try:
    rv=df_rv["vrad"][1:].to_numpy(float)
    err_rv=df_rv["svrad"][1:].to_numpy(float)
    tv=df_rv["rjd"][1:].to_numpy(float)
    #except:
        #return "No header name of the typr 'vrad' or 'svrad' found."

    return tv, rv, err_rv

def in_phase(t,T0=55528.86774,P=0.813475):
    E=(t-T0)/P

    phase =E%1.0

    phase=np.where(phase >0.5, phase-1, phase)
    return phase,E

def phase_fold(phase,objt, err_objt, Name=None):
    '''Phase fold of the Object passed'''

    plt.errorbar(phase, objt, err_objt, ls="")
    plt.xlabel("Phase")

    if Name != None:
        plt.ylabel(f"{Name}")
        plt.title(f"{Name} phase fold")

    else:
        plt.title("Phase fold")
    fname = (Name or "phase_fold").replace(" ", "_")
    plt.savefig(f"{fname}_phase_fold.png", dpi=200, bbox_inches="tight")
    plt.show()
    

def time_plt(time, objt, err_objt, Name=None):
    plt.errorbar(time, objt, err_objt, ls="")
    plt.xlabel("time")

    if Name != None:
        plt.ylabel(f"{Name}")
        plt.title(f"{Name} vs Time")

    else:
        plt.title("Time dependence")
    fname = (Name or "time_dep").replace(" ", "_")
    plt.savefig(f"{fname}_time_depend.png", dpi=200, bbox_inches="tight")
    plt.show()



def main():
    file_flux,file_rads=info()

    #Data acquisition
    t,flux,err_flux=file_opening(file_flux)
    tv,rv,err_rv = rad_vel(file_rads)
    #Clean the data
    t_cln,flux_cln,err_flux_cln = clean_subt_activity(t,flux,err_flux,sigma=3,interval=0.5)
    #tv_cln,rv_cln,err_rv_cln=clean_subt_activity(tv,rv,err_rv,sigma=3,interval=0.5)

    #Time dependence
    time_plt(t_cln,flux_cln,err_flux_cln,Name="Flux")
    time_plt(tv,rv,err_rv,Name="Radial velocity")

    #Phases
    phase_time,E_time=in_phase(t_cln)
    phase_rv,E_rv=in_phase(tv)
    
    #Phase fold graphs
    phase_fold(phase_time,flux_cln,err_flux_cln,Name="Flux")
    phase_fold(phase_rv, rv, err_rv,Name="Radial Velocity")


    #Periods
    epochs = np.unique(np.round(E_time).astype(int))
    T_t = T0 + epochs * P

    #Data analysed
    data_flux={"TIME":t_cln, "FLUX":flux_cln, "ERR_FLUX":err_flux_cln, "FLUX_PH":phase_time}
    data_rv={"rjd":tv, "vrad":rv, "svrad":err_rv, "RV_PH":phase_rv}
    data_tt={"T_t":T_t}

    df_flx=pd.DataFrame(data_flux)
    df_rv=pd.DataFrame(data_rv)
    df_Tt=pd.DataFrame(data_tt)

    df_flx.to_csv("lc_clean.csv", index=False)
    df_rv.to_csv("rv_clean.csv", index=False)
    df_Tt.to_csv("Periods",index=False)




if __name__ == "__main__":
     main()

