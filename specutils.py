# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 12:51:23 2014

@author: Parke
"""

import numpy as np
import my_numpy as mnp
from scipy.interpolate import interp1d
from math import ceil
import matplotlib.pyplot as plt
from scipy.signal import argrelmax

def coadd(wavelist, fluxlist, fluxerrlist, weightlist, masklist):
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
        
    masklist : list, optional
        A list of boolean arrays to mask out certain pixels. True values
        mean the pixel shouldn't be used.

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
    if masklist is None:
        masklist = [np.zeros(flux.shape, bool) for flux in fluxlist]
    
    lenvec = lambda lst: np.array(map(len, lst))
    wMs, fMs, feMs, mMs = map(lenvec, [wavelist, fluxlist, fluxerrlist, masklist])
    if any(fMs != feMs):
        raise ValueError('Corresponding flux and flux error vectors must be '
                         'the same lengths.')
    for weights, fM in zip(weightlist, fMs):
        try:
            wtM = len(weights)
            if wtM != fM:
                raise ValueError('All weights must either be floats or 1-D'
                                 'numpy arrays the same length as the '
                                 'corresponding flux array.')
        except TypeError:
            if type(weights) not in [float, int]:
                raise TypeError('Each elements of weightlist must either '
                                'be a float or a numpy array.')
    if any(np.logical_and(wMs != fMs, wMs != (fMs + 1))):
        raise ValueError('All wavelength arrays must have a length equal to '
                         'or one greater than the corresponding flux array.')         
    if any(fMs != mMs):
        raise ValueError('Corresponding flux and weight vectors must be the '
                         'same lengths.')
    
    #get wavelength edges for any wavelengths that are grids
    welist = []
    for wave, fM in zip(wavelist, fMs):
        we = mnp.mids2edges(wave) if len(wave) == fM else wave
        welist.append(we)
    
    #construct common grid
    w = common_grid(welist)
    
    #coadd the spectra (the m prefix stands for master)
    mfluence, mvar, mweight, mflux, merr = np.zeros([5, len(w)-1])
    #loop through each order of each x1d
    for we, flux, err, weight, mask in zip(welist, fluxlist, fluxerrlist, 
                                           weightlist, masklist):
        #intergolate and add flux onto the master grid
        dw = we[1:] - we[:-1]
        fluence, fluerr = flux*dw*weight, err*dw*weight
        fluence[mask], fluerr[mask] = np.nan, np.nan
        wrange = [np.min(we), np.max(we)]
        overlap = (np.digitize(w, wrange) == 1)
        wover = w[overlap]
        addflu = mnp.rebin(wover, we, fluence)
        addvar = mnp.rebin(wover, we, fluerr**2)
        addweight = mnp.rebin(w[overlap], we, weight)/(wover[1:] - wover[:-1])
        addmask = np.isnan(addflu)
        addflu[addmask], addvar[addmask], addweight[addmask] = 0.0, 0.0, 0.0
        i = np.nonzero(overlap)[0][:-1]
        mfluence[i] += addflu
        mvar[i] += addvar
        mweight[i] += addweight
    
    mdw = w[1:] - w[:-1]
    mmask = (mweight == 0.0)
    good = np.logical_not(mmask)
    mflux[good] = mfluence[good]/mweight[good]/mdw[good]
    merr[good] = np.sqrt(mvar[good])/mweight[good]/mdw[good]
    mflux[mmask], merr[mmask] = np.nan, np.nan
    return w, mflux, merr, mweight
    
def common_grid(wavelist):
    """Generates a common wavelength grid from any number of provided grids
    by using the lowest resolution grid wherever there is overlap.
    
    This is not a great method. Oversampling is still possible. It is fast
    though. 
    """
    
    #succesively add each grid to a master wavegrid
    #whereever the new one overlaps the old, pick whichever has fewer points
    we = wavelist[0]
    for wei in wavelist[1:]:
        #if no overlap, just app/prepend
        if we[-1] < wei[0]:
            we = np.append(we,wei)
            continue
        if we[0] > wei[-1]:
            we = np.append(wei,we)
            continue
            
        #identify where start and end of wei fall in we, and vise versa
        i0,i1 = np.digitize(wei[[0,-1]], we)
        j0,j1 = np.digitize(we[[0,-1]], wei)
        #we[i0:i1] is we overlap with wei, wei[j0:j1] is the opposite
        
        #pick whichever has fewer points (lower resolution) for the overlap
        Nwe, Nwei = i1-i0, j1-j0 #no of points for eachch in overlap
        if Nwe < Nwei:
            #get pieces of wei that fall outside of we
            wei_pre, _, wei_app = np.split(wei, [j0,j1])
            #stack with we. leave off endpts to avoid fractional bins at the
            #switchover points
            we = np.hstack([wei_pre[:-1], we, wei_app[1:]])
        else: #same deal
            we_pre, _, we_app = np.split(we, [i0,i1])
            we = np.hstack([we_pre[:-1], wei, we_app[1:]])
            
    return we
            
    
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
                             
def rebin(newedges, oldedges, flux, error):
    """
    Rebin a spectrum given the edges of the old and new wavelength grids.
    
    Assumes flux is per wavelength, but otherwise units are irrelevent so long
    as they are consistent.
    
    Returns flux,error for the new grid.
    """
    dwold = oldedges[1:] - oldedges[:-1]
    dwnew = newedges[1:] - newedges[:-1]
    
    intflux = flux*dwold
    interror = error*dwold
    newintflux = mnp.rebin(newedges, oldedges, intflux)
    newintvar = mnp.rebin(newedges, oldedges, interror**2)
    newinterror = mnp.sqrt(newintvar)
    return newintflux/dwnew, newinterror/dwnew
    