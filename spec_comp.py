# Compara o especrto de duas estrelas 

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys

from espectro import helpme

# <cript> <file1> <file2> <center> <range> <optional:normalize>
def valida():
    '''
    Checks user input and retrives information given
    '''

    if (len(sys.argv)) < 5:
        print("Not enough arguments were passed ! Use format <script> <file1> <file2> <center> <range> <optional:normalize>")
        sys.exit(1)

    elif (not sys.argv[1].endswith(".fits") ) or ( not sys.argv[2].endswith(".fits") ):
        print("Invalid file format. Please provide a FITS file.")
        sys.exit(1)

    else:
        file1_path=sys.argv[1]
        file2_path=sys.argv[2]
        center=float(sys.argv[3])
        rng=float(sys.argv[4])

        # Check if the normalize argument is provided and if is valid 
        if ( len(sys.argv) > 5 ) and ( sys.argv[5] in ["median", "max"] ):
            normalize = sys.argv[5]
        else:
            print("No valid normalization method provided. The flux values will not be normalized.")
            normalize = None #Returns default value

        return (file1_path,file2_path,center,rng,normalize)


def hdus(file_path1,file_path2):

    '''
    Extracts the wavelength and flux from the FITS files and returns them as numpy arrays
    '''

    data1 = fits.getdata(file_path1)
    data2 = fits.getdata(file_path2)

    #Check the way the wavelength is stored in the FITS file, as it can be either "WAVE" or "WAVELENGTH"
    try:
        flux1=np.array(data1["FLUX"])
        wave1=np.array(data1["WAVE"])

        flux2=np.array(data2["FLUX"])
        wave2=np.array(data2["WAVE"])
    except:
        flux1=np.array(data1["FLUX"])
        wave1=np.array(data1["WAVELENGTH"])

        flux2=np.array(data2["FLUX"])
        wave2=np.array(data2["WAVELENGTH"])

    return (wave1,flux1,wave2,flux2)

def plotting(wave1,flux1,wave2,flux2,center,rng=500,normalize=None):

    #Filters

    filt1 = (wave1 >= center - rng) & (wave1 <= center + rng)
    filt2 = (wave2 >= center - rng) & (wave2 <= center + rng)

    wave1_filt = wave1[filt1]
    flux1_filt = flux1[filt1]
    wave2_filt = wave2[filt2]
    flux2_filt = flux2[filt2]


    # Perform normalization if requested
    match normalize:
        case "median":
            median_flux1 = np.median(flux1_filt)
            flux1_filt = flux1_filt / median_flux1

            median_flux2 = np.median(flux2_filt)
            flux2_filt = flux2_filt / median_flux2
        case "max":
            max_flux1 = np.max(flux1_filt)
            flux1_filt = flux1_filt / max_flux1

            max_flux2 = np.max(flux2_filt)
            flux2_filt = flux2_filt / max_flux2 
            
        case "mean":
            mean_flux1 = np.mean(flux1_filt)
            flux1_filt = flux1_filt / mean_flux1

            mean_flux2 = np.mean(flux2_filt)
            flux2_filt = flux2_filt / mean_flux2

    # Plot the filtered spectra
    plt.figure(figsize=(10, 6))
    plt.plot(wave1_filt, flux1_filt, label='Spectrum 1')
    plt.plot(wave2_filt, flux2_filt, label='Spectrum 2')
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.title(f'Spectral Comparison around {center}')
    plt.legend()
    plt.show()

def main():
    #help fuction
    if len(sys.argv) > 1 and (sys.argv[1].startswith("helpme") or sys.argv[1].startswith("help")):
        helpme()
        sys.exit(1)

    (file1_path,file2_path,center,rng,normalize)=valida()
    wave1,flux1,wave2,flux2=hdus(file1_path,file2_path)
    plotting(wave1,flux1,wave2,flux2,center,rng,normalize)

if __name__=="__main__":
    main()