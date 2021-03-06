# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 12:51:23 2014

@author: Parke
"""

from __future__ import division, print_function, absolute_import
import numpy as np
from . import my_numpy as mnp
from . import statsutils as stats
from . import pdfutils as pu
import matplotlib.pyplot as plt
from .specutils_bkwd import *
from scipy.stats import skew
from functools import reduce


def coadd(wavelist, fluxlist, fluxerrlist, weightlist, flaglist=None, masklist=None, extras=None):
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

    weightlist : list of floats or array-likes
        The weight to be applied to each flux measurement. These can be single
        values to be applied to the entire flux vector or given for each pixel
        (i.e. of identical length to the flux vectors). The most common example
        is exposure time weighting, where weights would be a list of exposure
        times associated with each spectrum. It is up to the user to provide
        weights and flux units that are consistent. Using the exposure time
        example, the flux units should include a per unit time component for
        weighting by exposure time to provide a correct result.

    flaglist : list of array-like, optional
        Data quality flags to be propagated in a bitwise-or sense. Obviously,
        if the flags for different arrays mean different things this is
        pointless.

    masklist : list, optional
        A list of boolean arrays to mask out certain pixels. True values
        mean the pixel shouldn't be used.

    extras : 2xN list, optional
        A list of additional vectors to coadd. Each list entry is a length N
        list of vectors and a string speciyfing how they should be coadded.
        Acceptable coaddition methods are 'avg', 'sum', 'min', 'max', and 'or'.
        For example, to track the start date of the first observation and end
        date of the last observation contributing to each bin, use
        >>> extras = [[startdates, 'min'], [enddates, 'max']]
        where startdates and enddates contain N vectors, each giving the start
        and end dates for each bin in the N spectra.

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

    lenvec = lambda lst: np.array(list(map(len, lst)))
    wMs, fMs, feMs, mMs = list(map(lenvec, [wavelist, fluxlist, fluxerrlist, masklist]))
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

    elist = extras
    extras = extras is not None
    flags = flaglist is not None
    if flags:
        assert all(fMs == lenvec(flaglist))

    #get wavelength edges for any wavelengths that are grids
    welist = []
    for wave, fM in zip(wavelist, fMs):
        we = mnp.mids2edges(wave) if len(wave) == fM else wave
        welist.append(we)

    #construct common grid and get regions where this grid fully overlaps
    #the other grids (i.e. no partial bins)
    w = common_grid(welist)
    overlist, woverlist = list(zip(*[__woverlap(wi, w) for wi in welist]))

    #coadd the spectra (the m prefix stands for master)
    mfluence, mvar, mweight, mflux, merr = np.zeros([5, len(w)-1])
    if flags:
        mflags = np.zeros(len(w)-1, flaglist[0].dtype)

    #loop through each order of each x1d
    for i in range(len(welist)):
        #intergolate and add flux onto the master grid
        we, wover, flux = welist[i], woverlist[i], fluxlist[i]
        err, weight, mask = fluxerrlist[i], weightlist[i], masklist[i]
        overlap = overlist[i]
        dw = np.diff(we)
        fluence, fluerr = flux*dw*weight, err*dw*weight
        fluence[mask], fluerr[mask] = np.nan, np.nan
        addflu = mnp.rebin(wover, we, fluence, 'sum')
        addvar = mnp.rebin(wover, we, fluerr**2, 'sum')
        addweight = mnp.rebin(wover, we, weight*dw, 'sum')/np.diff(wover)
        addmask = np.isnan(addflu)
        addflu[addmask], addvar[addmask], addweight[addmask] = 0.0, 0.0, 0.0
        mfluence[overlap] = mfluence[overlap] +  addflu
        mvar[overlap] = mvar[overlap] + addvar
        mweight[overlap] = mweight[overlap] + addweight

        if flags:
            addflag = mnp.rebin(wover, we, flaglist[i], 'or')
            mflags[overlap] = mflags[overlap] | addflag

    if extras:
        mextras = []
        for method, values in elist:
            mextras.append(stack_special(welist, values, method, commongrid=w))

    mdw = np.diff(w)
    mmask = (mweight == 0.0)
    good = np.logical_not(mmask)
    mflux[good] = mfluence[good]/mweight[good]/mdw[good]
    merr[good] = np.sqrt(mvar[good])/mweight[good]/mdw[good]
    mflux[mmask], merr[mmask] = np.nan, np.nan

    result = [w, mflux, merr, mweight]
    if flags:
        result.append(mflags)
    if extras:
        result.append(mextras)
    return result


def common_grid(wavelist):
    """Generates a common wavelength grid from any number of provided grids
    by using the lowest resolution grid wherever there is overlap.

    This is not a great method. Oversampling is still possible. It is fast
    though.
    """
    wavelist = sorted(wavelist, key = lambda w: w[0])

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
        i0,i1 = np.searchsorted(we, wei[[0,-1]])
        j0,j1 = np.searchsorted(wei, we[[0,-1]])
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


def stack_special(wavelist, valuelist, function, commongrid=None, baseval=None):
    """
    Rebin the values to a common wavelength grid and apply the function to
    compute the stacked result.

    Parameters
    ----------
    wavelist : list
        List of array-likes giving the edges of the wavelength bins for each array
        of values.
    list : list
        List of array-likes with the values to be stacked.
    function : function or string
        Function to be applied to list resulting arrays to produce stacked
        values. Must accept array input and an axis keyword, such as
        numpy.min. A string can be used to specify 'sum', 'min', 'max', or 'or'.
    commongrid : 1D array-like, optional
        The grid to use for rebinning and min-stacking the values. If None,
        common_grid(wavelist) will be used.
    baseval : scalar, optional
        The value to use as the initializion value for the array of stacked
        values. For example, if arrays were being stacked in an "and" sense,
        then the user may wish to start with all bins having a value of
        baseval=True, which may be changed to false as each set of values is
        successively stacked.

    Returns
    -------
    commongrid : 1D array-like
        Edges of the grid onto which the values have been rebinned and stacked.
    stack : 1D array-like
        The min-stack of the values (length is one less than commongrid).
    """
    default_bases = {'avg':0, 'sum':0, 'min':np.inf, 'max':-np.inf, 'or':0}
    if baseval is None:
        baseval = default_bases[function]
    w = common_grid(wavelist) if commongrid is None else commongrid
    iover, wover = list(zip(*[__woverlap(wi, w) for wi in wavelist]))
    nullary = np.array([baseval]*(len(w)-1), np.result_type(*valuelist))

    rebinlist = []
    for io, wo, wi, vi in zip(iover, wover, wavelist, valuelist):
        rebinned = np.copy(nullary)
        rebinned[io] = mnp.rebin(wo, wi, vi, function)
        rebinlist.append(rebinned)

    if function == 'sum':
        return np.sum(rebinlist, axis=0)
    elif function == 'or':
        return reduce(np.bitwise_or, rebinlist)
    elif function == 'min':
        return np.min(rebinlist, axis=0)
    elif function == 'max':
        return np.max(rebinlist, axis=0)
    elif function == 'avg':
        return np.mean(rebinlist, axis=0)
    else:
        return function(rebinlist, axis=0)


def polyfit(wbins, y, order, err=None):
    """
    Fit a centered polynomial to spectral data.

    Parameters
    ----------
    wbins : array-like
        Nx2 array of wavelength bins (gaps okay).
    y : array-like
        len(N) array of spectral data, assumed to be per bin unit.
    order : int
        order of the polynomial to fit
    err : array-like or None
        errors of the specral data

    Returns
    -------
    coeffs : 1D array
        Coefficients of the fit.
    covar : 2D array
        Covariance matrix for the coefficients.
    fitfunc : function
        Function that evaluates y and yerr when given new bins using the
        polynomial fit.
    """
    wmid = (wbins[0, 0] + wbins[-1, 1]) / 2.0

    #integrate the data
    dw = wbins[:, 1] - wbins[:, 0]
    yy = y * dw
    if err is None:
        ee = np.ones(len(y))
    else:
        ee = err * dw

    # fit integrated data
    coeffs, covar, yyfun = mnp.polyfit_binned(wbins - wmid, yy, ee, order)

    # create function for rapidly computing spectral values from fit
    def fitfunc(wbins):
        dw = wbins[:, 1] - wbins[:, 0]
        yy, ee = yyfun(wbins - wmid)
        return yy / dw, ee / dw

    return coeffs, covar, fitfunc


def split(wbins, y, err=None, contcut=2.0, linecut=2.0, method='skewness',
          contfit=4, plotspec=False, maxiter=1000, silent=False):
    """
    Split a spectrum into its continuum, emission, absorption, and intermediate
    components.

    Parameters
    ----------
    wbins : 2D array-like, shape Nx2
        the left (wbins[:.0]) and right (wbins[:,1]) edges of the bins in the
        spectrum
    y : 1D array-like, length N
        spectrum data
    err : 1D array-like or None, length N
        error on the spectrum data. only used in fitting a polynomial at the
        moment, so can be None if type(trendfit) is not int.
    contfit : {int|function|str}, optional
        Trend to be fit and removed from the continuum data.

        int :
            order of a polynomial fit to be used
        function :
            user-defined function that returns the fit values at the
            data points when given a boolean array identifying which points to
            fit
        str :
            any of the built-in trends for mypy.statutils.clean. As of
            2015-04-07 these are 'mean' and 'median'.
    contcut : float, optional
        Limit for determing continuum data. If method == 'skewness', this is
        the acceptable probability that the skewness of the continuum could
        have resulted from an appropriate normal distribution. If method ==
        'area', it is the maximum allowable area of any feature for it to still
        be considered continuum.
    linecut : float, optional
        Limit for determining line data, similar to contcut. If method ==
        'skewness', this is the minimum probability below which data should be
        considered not a result of line emission or absorption. If method ==
        'area', it is the minimum area for a feature to be considered a line.
    method : {'skewness'|'skewtest'|'area'}, optional
        If 'skewness' use the absolute value of the skewness sample statistic.
        If 'skewtest' use the
        p-value of the skewness statistical test to separate continuum,
        line, and "unknown" data. If 'area', use the area of bumps above and
        below the continuum  fit to separate continuum from line.
    maxiter : int, optional
        Throw an error if this many iterations pass.
    plotspec : {False|True}, optional
        If True plot the spectrum color-coded by line, continuum, and unknown
        data.

    Returns
    -------
    flags : 1D int array, length N
        An array with values from 0-3 flagging the data as 0-unflagged
        (unknown/intermediate), 1-emission, 2-absorption, 3-continuum. Use the
        flags2ranges function to convert these to wavelength start and end
        values.

    Notes
    -----
    - Relies on the mypy.statsutils.clean function.
    """

    if err is None:
        err = np.ones_like(y)
    wbins, y, err = list(map(np.asarray, [wbins, y, err]))
    assert len(wbins) == len(y)
    assert wbins.shape[1] == 2
    assert method in ['area', 'skewness', 'skewtest']
    dw = wbins[:, 1] - wbins[:, 0]

    # if polynomial fit, make the appropriate contfit function
    if type(contfit) is int:
        # order of the polynomial
        polyorder = contfit

        # trend function
        def contfit(good):
            fun = polyfit(wbins[good], y[good], polyorder, err[good])[2]
            return fun(wbins)[0]

    # make metric that will compute area
    metric = lambda x: x * dw

    # choose appropriate test
    if method == 'area':
        test = 'deviation size'
    if method == 'skewtest':
        test = 'skew'
    if method == 'skewness':
        def test(x, good):
            return abs(skew(x[good]))

    # identify continuum
    cont = stats.clean(y, contcut, test, metric, contfit, maxiter=maxiter,
                       printsteps=(not silent))

    # subtract the trend fit through the continuum before identifying lines
    yfit = contfit(cont)
    y_detrended = y - yfit

    # identify lines
    lines = ~stats.clean(y_detrended, linecut, test, metric, None,
                         maxiter=maxiter)

    # make sure there is no overlap between lines and continuum
    if np.any(lines & cont):
        raise ValueError('Lines and continuum overlap. Consider a larger '
                         'difference between contlim and linelim values.')

    # make integer flag array
    flags = np.zeros(y.shape, 'i1')
    above = y_detrended > 0.0
    flags[lines & above] = 1 #emission
    flags[lines & ~above] = 2 #absoprtion
    flags[cont] = 3 #continuum

    # plot, if desired
    if plotspec:
        labels = ['none of the below', 'emission', 'absorption', 'continuum']
        labels = [labels[i] for i in np.unique(flags)]
        color_flags(wbins, y, flags, labels=labels)
        plot(wbins, yfit, '--', label='continuum fit')
        plt.legend()

    return flags


def adaptive_continuum(wbins, y, window, skew_test_pass=0.95, maxiter=1000):

    wbins, y = list(map(np.asarray, [wbins, y]))
    assert len(wbins) == len(y)
    assert wbins.shape[1] == 2
    if y.ndim > 1 and y.shape[1] == 1: y = y.T

    wmid = (wbins[:,0] + wbins[:,1])/2.0
    n = np.round((wmid[-1] - wmid[0]) / window)
    edges = np.linspace(wmid[0], wmid[-1], n)

    data = np.vstack((wmid, wbins.T, y))
    pieces = mnp.divvy(data, edges, 0)

    cont_pieces = []
    for piece in pieces:
        wbins = piece[1:3].T
        y = piece[3]

        dw = wbins[:,1] - wbins[:,0]

        #trend function
        # def contfit(good):
        #     fun = polyfit(wbins[good], y[good], 1)[2]
        #     return fun(wbins)[0]
        # def contfit(good):
        #     return np.mean(y[good])
        def contfit(good):
            return np.median(y[good])

        # make metric that will compute area
        metric = lambda x: x * dw

        # identify continuum
        try:
            cont = stats.clean(y, skew_test_pass, 'runs', metric, contfit, maxiter=maxiter)
        except:
            cont = np.zeros_like(y, bool)
        cont_pieces.append(cont)

    return np.hstack(cont_pieces)


def gapsplit(wbins, other_vecs=None):
    w0, w1 = wbins.T
    gaps = (w0[1:] > w1[:-1])
    isplit = list(np.nonzero(gaps)[0] + 1)
    isplit.insert(0, 0)
    isplit.append(None)
    splits = list(zip(isplit[:-1], isplit[1:]))
    splitbins =  [wbins[i0:i1] for i0,i1 in splits]
    if other_vecs is not None:
        splitvecs = [[a[i0:i1] for i0,i1 in splits] for a in other_vecs]
        return splitbins, splitvecs
    else:
        return splitbins


def flags2ranges(wbins, flags):
    """
    Identifies and returns the start and end wavelengths for consecutive runs
    of data that are flagged.

    Primarily intended for use with splitspec.

    Returns
    -------
    ranges : list of 2xN arrays
        start and end wavelengths for the runs of each flag value in sorted
        order (i.e. np.unique(flags))
    """
    splitbins, splitflags = gapsplit(wbins, [flags])
    if len(splitbins) > 1:
        rngslist = list(map(flags2ranges, splitbins, splitflags[0]))
        return np.vstack(rngslist)
    begs, ends = mnp.block_edges(flags)
    left = wbins[begs, 0]
    right = wbins[ends-1, 1]
    return np.vstack([left, right]).T


def rebin(newedges, oldedges, flux, error, flags=None):
    """
    Rebin a spectrum given the edges of the old and new wavelength grids.

    Assumes flux is per wavelength, but otherwise units are irrelevent so long
    as they are consistent.

    Returns flux,error for the new grid.
    """
    newedges, oldedges, flux, error = list(map(np.asarray, [newedges, oldedges, flux, error]))

    dwold = oldedges[1:] - oldedges[:-1]
    dwnew = newedges[1:] - newedges[:-1]

    intflux = flux*dwold
    interror = error*dwold
    newintflux = mnp.rebin(newedges, oldedges, intflux, 'sum')
    newintvar = mnp.rebin(newedges, oldedges, interror**2, 'sum')
    newinterror = np.sqrt(newintvar)
    result = [newintflux/dwnew, newinterror/dwnew]
    if flags is not None:
        flags = np.asarray(flags)
        newflags = mnp.rebin(newedges, oldedges, flags, 'or')
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
    err :
        error to plot as polygon
    ealpha :
        transparency of error polygon
    *kwargs :
        keyword arguments to be passed to plot

    Returns
    -------
    plts : list
        List of plot objects.
    """

    err = kwargs.pop('err', None)
    ealpha = kwargs.pop('ealpha', 0.15)

    #fill gaps with nan
    isgap = ~np.isclose(wbins[1:, 0], wbins[:-1, 1])
    gaps = np.nonzero(isgap)[0] + 1
    n = len(gaps)
    wbins = np.insert(wbins, gaps, np.ones([n, 2])*np.nan, 0)
    f = np.insert(f, gaps, [np.nan]*n)

    #make vectors that will plot as a stairstep
    w = mnp.lace(*wbins.T)
    ff = mnp.lace(f, f)

    #plot stairsteps
    ax = kwargs.pop('ax', plt.gca())
    p = ax.plot(w, ff, *args, **kwargs)[0]

    if err is not None:
        err = np.insert(err, gaps, [np.nan]*n)
        ee = mnp.lace(err, err)
        e = ax.fill_between(w, ff-ee, ff+ee, color=p.get_color(), alpha=ealpha, edgecolor='none')
        return p, e

    return p


def color_flags(wbins, y, flags, *args, **kwargs):
    """
    Plot flagged spectral data with a different color for each flag.

    Parameters
    ----------
    wbins : 2-D array-like
        Wavlength bin edges as an Nx2 array.
    f : 1-D array-like
        Spectral data to plot, len(f) == N.
    flags : 1-D array-like
        Flags identifying different types of data. len(flags) == N. An example
        is continuum, absorption, and emission.
    *args :
        arguments to be passed to plot
    labels : list
        Uses as keyword. Labels for each flag value, in order.
    **kwargs :
        keyword arguments to be passed to plot.

    Returns
    -------
    plts : list
        List of plot objects.
    """
    plts = []
    if 'labels' in kwargs:
        labels = kwargs['labels']
        del kwargs['labels']
    else:
        labels = None

    for i, fl in enumerate(np.unique(flags)):
        pts = (flags == fl)
        if np.any(pts):
            if labels is not None:
                kwargs['label'] = labels[i]
            else:
                kwargs['label'] = 'flag {}'.format(fl)
            plts.append(plot(wbins[pts,:], y[pts], *args, **kwargs))
    plt.legend()

    return plts


def wave_offset(wbinsa, fa, ea, wbinsb, fb, eb, offsets, cmp_range, return_logpdf=False, return_error=True):
    """
    Compute a best-fit wavelength offset between two spectra and give an approximate 1-sigma error

    Parameters
    ----------
    wbinsa
    fa
    wbinsb
    fb
    offsets : 1d array
        offsets to sample
    cmp_range: 2-element array
        range of speca over which to compare the spectra
    return_pdf: True|False
        if True, return the likelihood of each offset. I created this so I could average PDFs from the MUSCLES G130M
        spectra to get a max-likelihood offset from all at once
    return_error
        obvious

    Returns
    -------
    offset, error

    """
    offsets = np.asarray(offsets)

    min_rng = [max(wbinsa[0], wbinsb[0] + offsets.max()), min(wbinsa[-1], wbinsb[-1] + offsets.min())]
    if cmp_range[0] < min_rng[0] or cmp_range[-1] > min_rng[-1]:
        raise ValueError('The spectra do not have sufficient span to cover cmp_range for all offsets.')

    keepa = mnp.inranges(wbinsa, cmp_range, inclusive=[True, True])
    ikeepa, = np.nonzero(keepa)
    wbinsa = wbinsa[keepa]
    fa, ea = [a[ikeepa[:-1]] for a in [fa, ea]]

    def chi2(offset):
        woffb = wbinsb + offset
        _fb, _eb = rebin(wbinsa, woffb, fb, eb)
        terms = (_fb - fa)**2/(ea**2 + _eb**2)
        return np.sum(terms)

    chis = np.array(list(map(chi2, offsets)))
    chis -= chis.min()
    likes = np.exp(-chis)

    if return_logpdf:
        normfac = np.trapz(likes, offsets)
        return -chis - np.log(normfac)

    if return_error:
        off, off0, off1 = pu.confidence_interval([offsets, likes], return_xpeak=True, use_mean=use_mean)
        e0, e1 = off - off0, off1 - off
        e = (e0 + e1)/2.0
        if (e1 - e0)/e > 1.0:
            raise ValueError('Single 1-sigma error value is not a good approximation in this case.')
        return off, e

    return offsets[np.argmax(likes)]


def __woverlap(wa, wb):
    """Find the portion of wb that overlaps wa with no partial bins, then
    return the idices for the values in those bins (for the b vector) and
    the wavelengths of the overlapping bin edges for wb."""
    wrange = wa[[0,-1]]
    overlap = (np.searchsorted(wrange, wb) == 1)
    iover = np.nonzero(overlap)[0]
    if wb[iover[0] - 1] == wrange[0]:
        overlap[iover[0] - 1] = True
        iover = iover - 1
    else:
        iover = iover[:-1]
    wover = wb[overlap]
    return iover, wover
