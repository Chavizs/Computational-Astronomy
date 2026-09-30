
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import sys
from PyAstronomy.pyasl import crosscorrRV


# Default mask: ESPRESSO_G2.fits

def valida():

    '''
    Checks user input and retrives information given. 
    Also gets the data from the mask selected by the user  
    '''

    if (len(sys.argv)) < 5:
        print("Not enough arguments were passed ! Use format <script> <file> <mask> <rmin> <rmax> <optional:normalize>")
        sys.exit(1)

    elif not sys.argv[1].endswith(".fits") : 
        print("File type not supported! Use a .fits file")
        sys.exit(1)

    else:
        file_path=sys.argv[1]
        mask_path=sys.argv[2]
        rmin=float(sys.argv[3])
        rmax=float(sys.argv[4])

        # Check if the normalize argument is provided and if is valid 
        if ( len(sys.argv) > 4 ) and ( sys.argv[4] in ["median", "max"] ):
            normalize = sys.argv[4]
        else:
            print("No valid normalization method provided. The flux values will not be normalized.")
            normalize = None #Returns default value

        return (file_path,mask_path,rmin,rmax,normalize)


def hdus(file_path, mask_path="ESPRESSO_G2.fits"):

    '''
    Extracts the wavelength and flux from the FITS file and returns them as numpy arrays
    '''

    data_file = fits.getdata(file_path)
    data_mask= fits.getdata(mask_path)

    #Check the way the wavelength is stored in the FITS file, as it can be either "WAVE" or "WAVELENGTH"
    try:
        flux=np.array(data_file["FLUX"])
        wave=np.array(data_file["WAVE"])

        t_flux=np.array(data_mask["FLUX"]) #Template flux
        t_wave=np.array(data_mask["WAVE"]) #template wavelength
    except:
        flux=np.array(data_file["FLUX"])
        wave=np.array(data_file["WAVELENGTH"])

        t_flux=np.array(data_mask["FLUX"])
        t_wave=np.array(data_mask["WAVELENGTH"])
        

    return (wave,flux,t_wave,t_flux)
    

def ccf(wave,flux,t_wave,t_flux,rmin,rmax):
    '''
    Computes the Cross-Correlation Function (CCF) using the crosscorRV function from PyAstronomy.
    '''
    rv, ccf = crosscorrRV(wave,flux, t_wave, t_flux, rmin, rmax, drv=0.5)

    return (rv,ccf)



def main():
    
    (file_path,mask_path,rmin,rmax,normalize)=valida()

    #Data acquisition 
    (wave,flux,t_wave,t_flux)=hdus(file_path,mask_path)

    #CCF
    (rv,Ccf)=ccf(wave,flux,t_wave,t_flux,rmin,rmax)

    rv_ideal =rv[np.argmax(Ccf)] #PQ USAR GAUSSIANA?
    return rv_ideal

if __name__ == "__main__":
    main()