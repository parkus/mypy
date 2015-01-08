# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 12:51:23 2014

@author: Parke
"""

import numpy as np
import my_numpy as mnp
import statsutils as stats
import matplotlib.pyplot as plt
from specutils_bkwd import *

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
        dw = np.diff(we)
        fluence, fluerr = flux*dw*weight, err*dw*weight
        fluence[mask], fluerr[mask] = np.nan, np.nan
        wrange = we[[0,-1]]
        overlap = (np.digitize(w, wrange) == 1)
        wover = w[overlap]
        addflu = mnp.rebin(wover, we, fluence)
        addvar = mnp.rebin(wover, we, fluerr**2)
        addweight = mnp.rebin(wover, we, weight*dw)/np.diff(wover)
        addmask = np.isnan(addflu)
        addflu[addmask], addvar[addmask], addweight[addmask] = 0.0, 0.0, 0.0
        i = np.nonzero(overlap)[0][:-1]
        mfluence[i] = mfluence[i] +  addflu
        mvar[i] = mvar[i] + addvar
        mweight[i] = mweight[i] + addweight
    
    mdw = np.diff(w)
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
            
    
def split(wbins, y, err, trendfit=4, linecut=0.05, contcut=0.05, maxiter=1000,
          plotsteps=False):
    """
    Split a spectrum into its continuum, emission, absorption, and intermediate
    components.
    
    The function fits a trend to the data, then measures how much each run
    of point consecuatively above or below the trend depart from it using
    chi2. The run with the largest chi2 value is removed and a trend is again
    fitted to the remaining data. At each step, the remaining data are tested
    for consistency with a gaussian distribution using the Shapiro-Wilks
    test. Different p-values for this test determine
    which runs are identified as lines, which as continuum, and which as
    something in between (see parameters below for elaboration).
    
    Parameters
    ----------
    wbins : 2D array-like, shape 2xN
        the left (wbins[0]) and right (wbins[1]) edges of the bins in the 
        spectrum
    y : 1D array-like, length N
        spectrum data
    err : 1D array-like, length N
        error on the spectrum data
    trendfit : {int|function}, optional
        Trend to be fit and removed from the data.
        int : order of a polynomial fit to be used
        function : user-defined function that returns the fit values at the
            data points when given a boolean array identifying which points to
            fit
    linecut : float
        The p-value of the shapiro-wilks test at which to record the flagged
        runs as lines.
    contcut : float
        The p-value of the shapiro-wilks test at which to record the remaining
        (unflagged) data as continuum. Specifying a value that is greater
        than linecut means some data may be identified as belonging to neither
        line or continuum regions. This is the "intermediate" data.
    maxiter : int
        Throw an error if this many iterations pass.
    plotsteps : plot the flagged and unflagged data at each iteration
    
    Returns
    -------
    flags : 1D int array, length N
        An array with values from 0-3 flagging the data as 0-unflagged
        (intermediate), 1-emission, 2-absorption, 3-continuum. Use the
        flags2ranges function to convert these to wavelength start and end
        values.
    """
    if len(wbins) != len(y):
        raise ValueError('The shape of wbins must be [len(y), 2]. These '
                         'represent the edges of the wavelength bins over which '
                         'photons were counted (or flux was integrated).')
    wbins, y, err = map(np.asarray, [wbins, y, err])
    flags = np.zeros(y.shape, 'i1')
    
    #if polynomial fit, make the appropriate trendfit function
    if type(trendfit) is int:
        polyorder = trendfit
        def trendfit(good):
            w0 = (wbins[0,0] + wbins[-1,-1])/2.0
            _wbins, _y, _err = wbins[good,:], y[good], err[good]
            fun = mnp.polyfit_binned(_wbins - w0, _y, _err, polyorder)[2]
            return fun(wbins - w0)[0]
            
    #identify lines
    lines = stats.flag_anomalies(y, test='sw', tol=linecut, trendfit=trendfit,
                                 plotsteps=plotsteps, maxiter=maxiter)
    trend = trendfit(~lines)
    positive = y > trend 
    flags[lines & positive] = 1 #emission
    flags[lines & ~positive] = 2 #absoprtion
    
    #identify continuum
    #make new trendfit to handle only the retained data
    cullmap = np.nonzero(~lines)[0] #maps retained data into all data
    def newtrendfit(newgood):
        good = np.zeros(y.shape, bool)
        good[cullmap[newgood]] = True
        return trendfit(good)
    intmd = stats.flag_anomalies(y[~lines], test='sw', tol=contcut, 
                                 trendfit=newtrendfit, plotsteps=plotsteps, 
                                 maxiter=maxiter)
    flags[cullmap[~intmd]] = 3 #continuum
    
    return flags
    
def flags2ranges(wbins, flags):
    """
    Identifies and returns the start and end wavelengths for consecutive runs
    of data with the same flag values.
    
    Primarily intended for use with splitspec.
    
    Returns
    -------
    ranges : list of 2xN arrays
        start and end wavelengths for the runs of each flag value (0 to 
        max(flags))
    """
    ranges = []
    for i in range(np.max(flags)):
        edges = mnp.block_edges(flags == i)
        left = wbins[0, edges[:-1]]
        right = wbins[1, edges[1:]]
        ranges.append(np.vstack([left, right]))
    return ranges
                             
def rebin(newedges, oldedges, flux, error, flags=None):
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
    newinterror = np.sqrt(newintvar)
    result = [newintflux/dwnew, newinterror/dwnew]
    if flags is not None:
        newflags = mnp.rebin_or(newedges, oldedges, flags)
        result.append(newflags)
    return result
    
def plot(wbins, f, *args, **kwargs):
    """
    Plot a spectrum as a stairstep curve, preserving gaps in the data.
    
    Parameters
    ----------
    wbins : 2-D array-like
        Wavlength bin edges as an Nx2 array.
    f : 1-D arra-like
        Spectral data to plot, len(f) == N.
    *args :
        arguments to be passed to plot
    *kwargs :
        keyword arguments to be passed to plot. consider passing color to
        prevent different sections of the spectrum being plotted with different
        colors
    
    Returns
    -------
    plts : list
        List of plot objects.
    """
    #split the spectrum at any gaps
    isgap = ~np.isclose(wbins[0,1:], wbins[1,:-1])
    gaps = np.nonzero(isgap)[0] + 1
    wbinlist = np.split(wbins, gaps, 1)
    flist = np.split(f, gaps)
    
    plts = []
    for [w0, w1], f in zip(wbinlist, flist):        
        #make vectors that will plot as a stairstep
        w = mnp.lace(w0, w1)
        f = mnp.lace(f, f)
        
        #plot stairsteps
        plts.append(plt.plot(w, f, *args, **kwargs))
    
    return plts