# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 12:51:23 2014

@author: Parke
"""

import numpy as np
import my_numpy as mnp
from scipy.interpolate import interp1d
from math import ceil

def coadd(wavelist, fluxlist, fluxerrlist, weightlist):
    """Coadds spectra of differing wavlength ranges and resolutions according
    to the provided weight factors.
    
    Detailed Description
    --------------------
    coadd begins by generating a common wavelength grid to use for the coadded
    spectrum. Effectively, this is done by using the
    lowest common resolution between all the input spectra covering a given
    wavelength. Spectra are multiplied by the supplied weight factors, then
    rebinned onto this common grid. Spectral bins that fall on the edge of a
    bin on the common grid are divided according to the fraction of the bin
    that falls on each side of the edge. Note that this will result in a
    statistical dependence between adjacent values in the final coadded 
    spectrum. When multiple source spectrum bins fall in a single coadded
    spectrum bin, weights are averaged across the bin. After rebinning, fluxes
    and weights are summed. Fluxes are then divided by the summed weights
    in each bin.
    
    Error propogation is handled by squaring the flux errors, then treating
    them identically to the flux values (multiplying by weights, rebinning,
    and dividing by summed weights). The square root of the result is then
    taken to produce a final error estimate. This is the standard propogation
    of errors scheme for the summation of weighted measurements, assuming the
    error on the weights is negligible.
    
    Parameters
    ----------
    wavelist : length N list of variable length 1-D numpy arrays
        Wavelength grids for the associated flux vectors in fluxlist. Must
        have the same length of fluxlist. If a wavelength grid's corresponding
        flux vector has length M, then the wavelength grid must have a length
        of M or M+1. A length M grid is interpreted as the centeres of
        wavelength bins whereas a length M+1 grid is interpreted as the edges
        of the wavelength bins. If centers are given, the bin edges will be
        reconstructed assuming a linear variation in bin width with wavelength.
        The units for each grid must be the same (e.g. nm, microns).

    fluxlist : length N list of variable length 1-D numpy arrays
        Flux vectors corresponding to the wavelength grids. Units must be per
        unit wavelength, matching the wavelength units used in wavelist. Units
        must also be identical for all vectors. 
    
    fluxerrlist : length N list of variable length 1-D numpy arrays
        Flux errors corresponding to the vectors in fluxlist.
        
    weights : length N list of scakars or variable length 1-D numpy arrays
        The weight to be applied to each flux measurement. These can be single
        values to be applied to the entire flux vector or given for each pixel
        (i.e. of identical length to the flux vectors). The most common example
        is exposure time weighting, where weights would be a list of exposure
        times associated with each spectrum. It is up to the user to provide 
        weights and flux units that are consistent. Using the exposure time
        example, the flux units should include a per unit time component for
        weighting by exposure time to provide a correct result. 

    Returns
    -------
        wavelength : a 1-D numpy array where len(wavelength) = len(flux) + 1
            The bin edges of the common wavelength grid generated for the
            coaddition.
            
        flux : a 1-D numpy array
            The coadded flux values in the same units as the input fluxes
            
        flux err : a 1-D numpy array
            Errors of the coadded flux values.
            
        weightsums : a 1-D numpy array
            The sum of the weights of the flux values added in each wavelength
            bin. For example, if exposure times were used as weights, this
            would be the total exposure time corresponding to the flux
            measurement in each bin.
    """
    
    #vet the input
    lenvec = lambda lst: np.array(map(len, lst))
    wMs, fMs, feMs = map(lenvec, [wavelist, fluxlist, fluxerrlist])
    if any(fMs != feMs):
        raise ValueError('Corresponding flux and flux error vectors must be ',
                         'the same lengths.')
    for weights, fM in zip(weightlist, fMs):
        try:
            wtM = len(weights)
            if wtM != fM:
                raise ValueError('All weights must either be floats or 1-D',
                                 'numpy arrays the same length as the ',
                                 'corresponding flux array.')
        except TypeError:
            if type(weights) not in [float, int]:
                raise TypeError('Each elements of weightlist must either ',
                                'be a float or a numpy array.')
    if any(np.logical_and(wMs != fMs, wMs != (fMs + 1))):
        raise ValueError('All wavelength arrays must have a length equal to ',
                         'or one greater than the corresponding flux array.')
    
    #get wavelength edges for any wavelengths that are grids
    welist = []
    for wave, fM in zip(wavelist, fMs):
        we = mnp.mids2edges(wave) if len(wave) == fM else wave
        welist.append(we)
    
    #construct common grid
    w = common_grid(welist)
    
    #coadd the spectra (the m prefix stands for master)
    mins, mvar, mexptime = [np.zeros(n-1)for i in range(3)]
    #loop through each order of each x1d
    for x1d in x1ds:
        flux_arr, err_arr, dq_arr = [x1d[1].data[s] for s in 
                                     ['flux', 'error', 'dq']]
        we_arr = wave_edges(x1d)
        exptime = x1d[1].header['exptime']
        for flux, err, dq, we in zip(flux_arr, err_arr, dq_arr, we_arr):
            #intergolate and add flux onto the master grid
            dw = we[1:] - we[:-1]
            flux, err = flux*dw, err*dw
            wrange = [np.min(we), np.max(we)]
            badpix = (dq != 0)
            flux[badpix], err[badpix] = np.nan, np.nan
            overlap = (np.digitize(w, wrange) == 1)
            addins = exptime*mnp.rebin(w[overlap], we, flux)
            addvar = exptime**2*mnp.rebin(w[overlap], we, err**2)
            
            addtime = np.ones(addins.shape)*exptime
            badbins = np.isnan(addins)
            addins[badbins], addvar[badbins], addtime[badbins] = 0.0, 0.0, 0.0
            i = np.nonzero(overlap)[0][:-1]
            mins[i] += addins
            mvar[i] += addvar
            mexptime[i] += addtime
    
    mdw = w[1:] - w[:-1]
    mflux = mins/mexptime/mdw
    merr = np.sqrt(mvar)/mexptime/mdw
    badbins = (mexptime == 0.0)
    mflux[badbins], merr[badbins] = np.nan, np.nan
    return w, mflux, merr, mexptime
    
def common_grid(wavelist):
    """Generates a common wavelength grid from any number of provided grids
    by employing the lowest resolution of the provided grids covering a given
    wavelength.
    """
    
    #get the edges, centers, and spacings for each wavegrid
    welist = wavelist
    dwlist = [we[1:] - we[:-1] for we in welist]
    wclist = map(mnp.midpts, welist)
    wmin, wmax = np.min(welist), np.max(welist)
    
    #use the central wavelength values to order the spacings into a single vector
    dw_all, wc_all = dwlist[0], wclist[0]
    for dw, wc in zip(dwlist[1:], wclist[1:]):
        i = np.digitize(wc, wc_all)
        dw_all = np.insert(dw_all, i, dw)
        wc_all = np.insert(wc_all, i, wc)
        
    #identify the relative maxima in the spacings vector and make an 
    #interpolation function between them
    _,imax = mnp.argextrema(dw_all)
    wcint = np.hstack([[wmin],wc_all[imax],[wmax]])
    dwint = dw_all[np.hstack([[0],imax,[-1]])]
    dwf = interp1d(wcint, dwint)
    
    #construct a vector by beginning at wmin and adding the dw amount specified
    #by the interpolation of the maxima
    w = np.zeros(ceil((wmax-wmin)/np.min(dw)))
    w[0] = wmin
    n = 1
    while True:
        w[n] = w[n-1] + dwf(w[n-1])
        if w[n] > wmax: break
        n += 1
    w = w[:n]
    
def identify_continuum(wbins, y, err, function_generator, maxsig=2.0, 
                       emission_weight=1.0, maxcull=0.99, plotsteps=False):
    #TODO: modify this to use mypy.stats.flag_anomalies
    if type(plotsteps) == bool:
        plotbelow = np.inf if plotsteps else 0.0
    else:
        plotbelow = plotsteps
        plotsteps = True
    if plotsteps: plt.ioff()
    if len(wbins) != len(y):
        raise ValueError('The shape of wbins must be [len(y), 2]. These '
                         'represent the edges of the wavelength bins over which '
                         'photons were counted (or flux was integrated).')
    wbins, y, err = map(np.array, [wbins, y, err])
    Npts = len(y)
    
    if plotsteps:
        wave = (wbins[:,0] + wbins[:,1])/2.0
        wbinsin, wavein, yin= wbins, wave, y
        
    while True:
        #fit to the retained data
        f = function_generator(wbins, y, err)
        
        #count the runs
        expected = f(wbins)
        posneg = (y > expected)
        run_edges = ((posneg[1:] - posneg[:-1]) !=0)
        Nruns = np.sum(run_edges) + 1
        
        #compute the PTE for the runs test
        N = len(y)
        Npos = np.sum(posneg)
        Nneg = N - Npos
        mu = 2*Npos*Nneg/N + 1
        var = (mu-1)*(mu-2)/(N-1)
        sigruns = abs(Nruns - mu)/np.sqrt(var)
        
        #if the fit passes the runs test, then return the good wavelengths
        if sigruns < maxsig:
            non_repeats = (wbins[:-1,1] != wbins[1:,0])
            w0, w1 = wbins[1:,0][non_repeats], wbins[:-1,1][non_repeats]
            w0, w1 = np.insert(w0, 0, wbins[0,0]), np.append(w1, wbins[-1,1])
            return np.array([w0,w1]).T
        else:
            #compute the chi2 PTE for each run
            iedges = np.concatenate([[0], np.nonzero(run_edges)[0]+1, [len(run_edges)+1]]) #the slice index
            chiterms = ((y - expected)/err)**2
            chisum = np.cumsum(chiterms)
            chiedges = np.insert(chisum[iedges[1:]-1], 0, 0.0)
            chis =  chiedges[1:] - chiedges[:-1]
            DOFs = (iedges[1:] - iedges[:-1])
            sigs = abs(chis - DOFs)/np.sqrt(2*DOFs)
            if emission_weight != 1.0:
                if posneg[0] > 0: sigs[::2] = sigs[::2]*emission_weight
                else: sigs[1::2] = sigs[1::2]*emission_weight
            
            good = np.ones(len(y), bool)            
            good[np.argmax(sigs)] = False #mask out the run with the smallest PTE
            keep = np.concatenate([[g]*d for g,d in zip(good, DOFs)]) #make boolean vector
            
            if plotsteps and (sigruns < plotbelow):
                plt.plot(wavein, yin)
                plt.plot(wavein,f(wbinsin),'k')
                plt.plot(wave,y,'g.')
                trash = np.logical_not(keep)
                plt.plot(wave[trash], y[trash], 'r.') 
                ax = plt.gca()
                plt.text(0.8, 0.9, '{}'.format(sigruns), transform=ax.transAxes)
                plt.show()
            if plotsteps: wave = wave[keep]
            wbins, y, err = wbins[keep], y[keep], err[keep] 
        
        if float(len(y))/Npts < (1.0 - maxcull):
            raise ValueError('More than maxcull={}% of the data has been '
                             'removed, but the remaining data and associated '
                             'fit is still not passing the Runs Test. Consider '
                             'checking that the fits are good, relaxing '
                             '(increasing) the maxsig condition for passing '
                             'the Runs Test, or increasing maxcull to allow '
                             'more data to be masked out.')