# Author Susana Barros , Jan 2019
#

import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from astropy.stats import sigma_clipped_stats




def remove_toutliears(time,flux,fluxerr, sigma):

    robustmean, robustmedian ,  robustrms  = sigma_clipped_stats(flux, sigma=sigma, maxiters=10)


    cleanlow = np.where((flux-robustmean)/robustrms < -sigma)[0]

    diff = -(cleanlow - np.roll(cleanlow,-1) )
    diff2 = (cleanlow - np.roll(cleanlow, 1) )
    flux4 = flux.copy()

    for i in range(0, len(cleanlow)):
        #if (diff[i] != 1) and (diff2[i] != 1):
        if    np.logical_and( np.not_equal(diff[i],1) , np.not_equal(diff2[i], 1)):
            flux4[cleanlow[i]] = 0

    #print(np.where(flux4 == 0)[0])
    clean = np.where( np.logical_and( np.less((flux-robustmean)/robustrms, sigma), np.not_equal(flux4, 0)))[0]

    #remove points
    time_clean=time[clean]
    flux_clean=flux[clean]
    fluxerr_clean=fluxerr[clean]


    return time_clean, flux_clean, fluxerr_clean





def filter_spline(time,flux,fluxerr, sigma, interval):

    # compute how many knots
    xspacedim= np.fix( (time[-1]-time[0])  /interval) -1
    # create the knots array
    xss = (np.arange(xspacedim)+1) * interval + time[0]


    # be careful with big gaps otherwise it fails maybe need to adjust the interval * 3
    diff = time - np.roll(time,-1)
    gap = np.where(diff < -interval*3)[0]

    if len(gap) > 0 :
        print('There is a big gap',time[gap[0]+1]-time[gap[0]] )
        flag2  = np.where( np.logical_or( np.less(xss, time[gap[0]]), np.greater(xss, time[gap[0]+1] )  ))
        xs = xss[flag2]
    else :
        xs=xss

    # lcq adds knots in both ends so if we dont cut them they can be too big
    #xs = xss[1:-1]

    robustmean, robustmedian ,  robustrms  = sigma_clipped_stats(flux, sigma=sigma, maxiters=10)


    # evaluate the spline at the times


    nt =len(time)
    nloop = 0
    number1 = 0
    flag = np.arange(nt)

    while nloop < 10:


        spl = LSQUnivariateSpline(time, flux, xs, k=3)
        interpola2 = spl(time)

        diff = interpola2 -flux
        flag = np.where( np.abs(diff/robustrms) < sigma)[0]
        number2 = len(flag)


        if flag[number2-1] != nt-1 :
            flag = np.append(flag, nt-1)
            interpola2[nt-1]= np.median(interpola2)

        if number2 == number1:
             nloop=20

        number1 = number2
        nloop = nloop+1.




    flux_filtered = flux / interpola2
    fluxerr_filtered =  fluxerr/interpola2


    return flux_filtered, fluxerr_filtered



def clean_subt_activity(time,flux,fluxerr, sigma, interval):

    robustmean1, robustmedian1 ,  robustrms1  = sigma_clipped_stats(flux, sigma=sigma, maxiters=10)


    flux_filtered, fluxerr_filtered = filter_spline(time,flux,fluxerr, 3, interval)

    robustmean, robustmedian ,  robustrms  = sigma_clipped_stats(flux_filtered, sigma=sigma, maxiters=10)

    if  robustrms  > robustrms1 :
        print('error in the splinefit')
        flux_filtered = flux
        fluxerr_filtered  = fluxerr

    time_clean, flux_clean, fluxerr_clean = remove_toutliears(  time,flux_filtered, fluxerr_filtered , sigma)


    return time_clean, flux_clean, fluxerr_clean
