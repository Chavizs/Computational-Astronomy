# Computational-Astronomy
Repository fully dedicated to Computational Astronomy projects 


Lines Selected:

lines={6173.34:3, 6358.68:3, 6380.75:3, 5987.07:3, 5775.08:3, 5787.92:3, 
       5956.70:3, 5916.26:3, 6481.88:3, 5054.65:2, 5905.68:3, 
       6027.06:3, 5965.84:3, 5978.55:3, 5977:3, 5727:3,
       6721.85:3,
        5543.94:1, 5010.94:1, 5688.22:1, 5711.09:1, 5690.43:1, 5772.15:1, 5948.54:1, 6142.49:1, 6145.02:1, 6237.33:1 , 6243.82:1, 6244.48:1, 6527.21:1, 
        5867.56:1, 6166.44:1, 6449.82:1, 6455.6:1, 6471.67:1, 6499.65:1, 5526.82:1, 6245.62:1, 6604.6:1, 4645.19:1, 4820.41:1, 
        5039.96:1, 5866.45:1, 6258.11:1, 5381.03:1, 5418.77:1, 4708.02:1, 5296.7:1, 5010.94:1, 5578.72:1, 5805.22:1, 
        6086.29:1, 6108.12:1, 6128.98:1, 6175.37:1, 6186.72:1, 6378.26:1, 6635.13:1, 6767.78:1, 6772.32:1, 4602.0:1, 4602.95:1, 
        4620.51:1,  4625.05:1, 4629.34:1, 4630.12:1, 4772.82:1, 4788.76:1,6154.23:1, 6160.75:1, 6318.72:1,  6195.46:1, 6721.85:1, 
        5016.17:1, 4739.11:1, 4626.18:1 , 5342.71:1, 5081.11:1,  5847.0:1, 6111.08:1,  6176.82:1,  6842.04:1, 4508.28:1, 
        4523.4:1, 4561.41:1,  4576.34:1, 4620.51:1 , 4757.58:1, 
        4802.52:2, 4802.88:2, 4946.39:1, 4962.58:2,  4994.14:2, 5044.22:2, 5054.65:2, 5067.15:2, 5068.77:2, 5074.76:1, 
        5090.78:2,  5127.36:2,  5159.06:2, 5180.06:2, 5198.72:2,  5229.86:2, 5242.50:2, 5243.78:2,   5253.47:2, 
        5288.53:2, 5373.71:2, 5379.58:2,  5389.48:2, 5417.04:2 , 5441.34:2,  5522.45:2 , 5560.22:2,  5633.95:2,  5638.27:2, 
        5653.87:2 ,5667.52:1 ,  5679.03:2, 5701.55:2 , 5715.09:1 , 5731.77:2 ,  5741.85:2, 5752.04:2,  5775.08:2,  5778.46:2, 
        5806.73:2, 5809.22:2,  5852.22:2, 5856.09:2 , 5862.36:2, 5905.68:2, 5916.26:2, 5927.79:2, 5934.66:2, 5956.70:2, 5983.69:2, 
        5984.82:2, 5987.07:2, 6005.55:2, 6024.06:2, 6027.06:2, 6056.01:2, 6065.49:2, 6082.72:2, 6096.67:2, 6127.91:2, 6157.73:2 }

STEEW
Stelar Temperature Estimator by Equivalent Width

The main goal of this script is to estimate the effective temperature of solar type stars. This project was developed in the contex of the Computacional Astronomy coursework from the Bachelor’s degree  in Physics.

Required Files

In order for this script to work properly, the user must have the calibation document (Calibration.csv & Calibration_CV.csv) , the HARPS template ( created by profesor Sergio ) and the .fits documnet(s) to be analised, all in the same folder. It is expected that the user will only use spectrums collected by HARPS. If the user opts to use anyother spectrum collected by other istrument the radial velocity correction will be poorly done , resulting in a spectrum that still contains strong traces of the Doopler effect.

Note: Up to now, this script only analyses one star at time. That is, only one .fits document is accepeted as input.

Structure 

In this project, the equivalent widths are calculated by fitting an gaussian to the spectral lines defined (such is done by the ew_gauss function)
As will be discussed later, the ew_gauss funtion only evaluates isolated specral lines. 
Therefore, the following python dictionary contais all the spectral lines identified as aproximatly isolated. The keys are the spectral lines and the values are the range used in the gaussian fit.


In order to correct the Doopler effect it is used ARSE's  Template for HARPS' spectrums

======================================
   Description of the methods used
======================================

- EW.py

This python fyle containing all functions used to compute the the equivalent width of a isolated line.

Inside it, one finds the function used to: 
. correct the spectrum of the star due to the Doopler effect (ccf) ;  
. normalizes the spectral region of the isolated absroption line ; 
. function used to estimate its equivalent width by fitting a gaussian.


. Normalization Function 

This process is based in a linear normalization (fitting a line thought the spectrum)
This function divides the spectral window centered at the line in 30 intervals (in an uniform fashion). It is calculated the standert deviation of each section and only 4 (2 in each end of the spectal window provided) out of the 30 initial sections are used to perform the linear fit. This linear fit will be used as the line of maximum values for the flux at each wavelength.

Return the flux normalized

 . ccf Function

 This function is responsible for correcting the radial velocity of the star. It uses the CrossCorrRV function from PyAstronomy. 
 It requires : 
 - (Optional) wave, flux in the region to be analysed
 - (!) t_wave , t_flux : wavelength and flux from the template model
 - (Optional) rv_min, rv_max : minimum and maximum radial velocity interval
 - (Optional) drv: step for the radial velocity

Returns an array with the correct values of wavelength

 . ew_fit Function

This function is responsible to perform the gaussian fit of each the spectral line.

Arguments:
- center of the absorption line
- size of the window (region of the spectrum around the line)
- wave , flux arrays Corrected from Doopler effect and normalized

It uses LMFit , however, it has a fallback method built in. That is, if the user doesn't have LMFit, it will use polyfit from numpy to perform the gaussian fit.
