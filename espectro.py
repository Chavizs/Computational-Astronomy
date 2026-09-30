
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import sys

# Formato do argumento:<script> <file> <center> <range> <normalize>

def helpme():

    '''
    Prints help messages to the user
    '''

    if sys.argv[1] == "helpme":
        print("Usage: <script> <file> <center> <range>")
        print("Arguments:")
        print("  <file>     : FITS file containing the spectral data - Phase 3")
        print("  <center>   : Central wavelength of the region to display")
        print("  <range>    : Range of wavelengths to display\n")
        print("To get more detailed information about the arguemnts, please use helpme_<argument>")
    
    else:
        match sys.argv[1]:
            case "helpme_file":
                print("The <file> argument should be a FITS file containing the spectral data. The file should have the following structure:")
                print("  - A primary HDU containing the header information")
                print("  - A binary table HDU containing the spectral data, with columns for wavelength and flux")
                print("The file should be in the format of Phase 3.")

            case "helpme_center":
                print("The <center> argument specifies the central wavelength of the region to display. It should be a floating-point number representing the wavelength in Angstroms.")
                print("The default <center> is 6562.8 Angstroms (H-alpha line)")

            case "helpme_range":
                print("The <range> argument specifies the range of wavelengths to display around the central wavelength. It should be a floating-point number representing the range in Angstroms.")
                print("The default <range> is 500 data points")

            case "helpme_normalize":
                print("The <normalize> argument specifies whether to normalize the flux values. It should be one of the following:")
                print("  - median: Normalize the flux values using the median value of the flux in the specified range as reference.")
                print("  - max: Normalize the flux values using the maximum value of the flux in the specified range as reference.")
                print("The default <normalize> is None. That is, the flux values will not be normalized and the argument can be omitted.")

            case _:
                print("Invalid help argument. Use 'helpme' for general usage or 'helpme_<argument>' for specific argument help.\nExemples: helpme_file")
        
        
        
def valida():

    '''
    Checks user input and retrives information given
    '''

    if (len(sys.argv)) < 4:
        print("Not enough arguments were passed ! Use format <script> <file> <center> <range> <optional:normalize>")
        sys.exit(1)

    elif not sys.argv[1].endswith(".fits") : 
        print("File type not supported! Use a .fits file")
        sys.exit(1)

    else:
        file_path=sys.argv[1]
        center=float(sys.argv[2])
        rng=float(sys.argv[3])

        # Check if the normalize argument is provided and if is valid 
        if ( len(sys.argv) > 4 ) and ( sys.argv[4] in ["median", "max"] ):
            normalize = sys.argv[4]
        else:
            print("No valid normalization method provided. The flux values will not be normalized.")
            normalize = None #Returns default value

        return (file_path,center,rng,normalize)
    

def hdus(file_path):

    '''
    Extracts the wavelength and flux from the FITS file and returns them as numpy arrays
    '''

    data = fits.getdata(file_path)

    #Check the way the wavelength is stored in the FITS file, as it can be either "WAVE" or "WAVELENGTH"
    try:
        flux=np.array(data["FLUX"])
        wave=np.array(data["WAVE"])
    except:
        flux=np.array(data["FLUX"])
        wave=np.array(data["WAVELENGTH"])

    return (wave,flux)


def plotting(wave,flux,center,rng=500,normalize=None):

    '''
    Plots the spectrum using matplotlib, with the option to normalize the flux values provided by the user.
    '''
    # mask to filter the data, so the script doesn't try to plot thousands of points
    flit= (wave > center - rng/2) & (wave < center + rng/2) 

    wave_filt=wave[flit]
    flux_filt=flux[flit]

    #Normalization of the flux selected
    match normalize:
        case "median":
            median_flux = np.median(flux_filt)
            flux_filt = flux_filt / median_flux

        case "max":
            max_flux = np.max(flux_filt)
            flux_filt = flux_filt / max_flux

        case None:  
            pass  # No normalization applied

    #Plotting the spectrum
    plt.plot(wave_filt,flux_filt)
    plt.xlim(center - rng/2 , center + rng/2)

    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    plt.title(f'Spectral Comparison around {center}')
    plt.legend()
    plt.show()
    plt.show()


def main():
    #help fuction
    if len(sys.argv) > 1 and (sys.argv[1].startswith("helpme") or sys.argv[1].startswith("help")):
        helpme()
        sys.exit(0)

    #plotting the spectrum
    else:
        (file_path,center,rng,normalize)=valida()
        (wave,flux)=hdus(file_path)
        plotting(wave,flux,center,rng,normalize)



if __name__=="__main__":
    main()


