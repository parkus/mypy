# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 11:43:52 2014

@author: Parke
"""
import numpy as np
from scipy.interpolate import interp1d #InterpolatedUnivariateSpline as ius #pchip_interpolate
import matplotlib.pyplot as plt
import warnings

def bracket(a, v):
    """
    return the indices of the two values in vector a that would bracket value v
    """
    match = (a == v)
    if sum(match):
        return np.nonzero(match)[0]
    else:
        i = np.searchsorted(a, v)
        if i == 0 or i == len(a):
            raise ValueError('Outside of phoenix grid range.')
        else:
            return [i-1, i]

def sliminterpN(pt, grids, datafunc):
    """
    A means of computing an N-linear interpolation without having to load a
    huge N-d data array into memory.
    
    Initially written for use in interpolating spectra from the a model spectra
    database.
    
    Parameters
    ----------
    pt : 1-D array-like
        The value of the point where interpolation of the data is to occur.
    grids : list of 1-D array-like objects
        The grids over which data is available, in the same order as the values
        in pt.
    datafunc : function
        A function that will return the data when the indices of the grid values are input
        (separately, in order -- not as a list). Data
        can be anything that permits linear comibination with normal arithmetic
        operators.
    
    Returns 
    -------
    result : same type as returned by datafunc
        The data interpolated to pt.
    """     
    #function to return interpolated value one level down the hierarchy
    def idata(i):
        #if we are down to one dimension, return the data
        if len(pt) == 1:
            return datafunc(i)
        
        else:
            newfunc = lambda *args: datafunc(i, *args)
            return sliminterpN(pt[1:], grids[1:], newfunc)
    
    #retrieve the bracketing values in the grid
    bkti = bracket(grids[0], pt[0])
    
    #if pt[0] happened to fall right on a grid value, use that data
    if len(bkti) == 1:
        return idata(bkti[0])
    
    #compute the factors by which the data from each point on the grid will
    #be mutliplied before summing
    a = grids[0][bkti]
    d = a[1] - a[0]
    fac0 = (a[1] - pt[0])/d
    fac1 = 1.0 - fac0
    
    #get the data at the bracketing values, interpolated over the other grid
    #values
    bktdata = map(idata, bkti)
    
    #interpolate
    return bktdata[0]*fac0 + bktdata[1]*fac1

def quadsum(*args, **kwargs):
    """Sum of array elements in quadrature.
    
    This function is identical to numpy.sum except that array elements are
    squared before summing and then the sqrt of the resulting sums is returned.
    
    The docstring from numpy.sum is reproduced below for convenience (copied 
    2014-12-09)
    
    Parameters
    ----------
    a : array_like
        Elements to sum.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.
        The default (`axis` = `None`) is perform a sum over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.

        .. versionadded:: 1.7.0

        If this is a tuple of ints, a sum is performed on multiple
        axes, instead of a single axis or all the axes as before.
    dtype : dtype, optional
        The type of the returned array and of the accumulator in which
        the elements are summed.  By default, the dtype of `a` is used.
        An exception is when `a` has an integer type with less precision
        than the default platform integer.  In that case, the default
        platform integer is used instead.
    out : ndarray, optional
        Array into which the output is placed.  By default, a new array is
        created.  If `out` is given, it must be of the appropriate shape
        (the shape of `a` with `axis` removed, i.e.,
        ``numpy.delete(a.shape, axis)``).  Its type is preserved. See
        `doc.ufuncs` (Section "Output arguments") for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    sum_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
        is returned.  If an output array is specified, a reference to
        `out` is returned.

    See Also
    --------
    ndarray.sum : Equivalent method.

    cumsum : Cumulative sum of array elements.

    trapz : Integration of array values using the composite trapezoidal rule.

    mean, average

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    Examples
    --------
    >>> np.sum([0.5, 1.5])
    2.0
    >>> np.sum([0.5, 0.7, 0.2, 1.5], dtype=np.int32)
    1
    >>> np.sum([[0, 1], [0, 5]])
    6
    >>> np.sum([[0, 1], [0, 5]], axis=0)
    array([0, 6])
    >>> np.sum([[0, 1], [0, 5]], axis=1)
    array([1, 5])

    If the accumulator is too small, overflow occurs:

    >>> np.ones(128, dtype=np.int8).sum(dtype=np.int8)
    -128
    """
    args = list(args)
    args[0] = args[0]**2
    return np.sqrt(np.sum(*args, **kwargs))

def lace(a, b, axis=0):
    """
    Combine the two arrays by alternating values from each.
    
    Parameters
    ----------
    a,b : array-like
        The two arrays to be laced together. Every dimension except that of
        axis must match. If the lengths along axis differ, the lacing will
        proceed from element 0 until one of the arrays ends, then the other
        array will be used for the remainder.
    axis : int, optional
        The axis along which to lace.
    
    Returns
    -------
    c : array
        The laced arrays, with the same shape of the input arrays except along
        the lace axis, which will be the sum of the lengths of the two input
        arrays along that axis.
    """    
    a, b = map(np.asarray, [a,b])
    
    #prepare empty output array
    Na, Nb = a.shape[axis], b.shape[axis]
    ctype = np.result_type(a,b)
    cshape = list(a.shape)
    cshape[axis] = Na + Nb
    c = np.zeros(cshape, ctype)
    
    #make the lace axis the first one
    a, b, c = [np.swapaxes(ary, axis, 0) for ary  in [a,b,c]]
    
    #decide how to interleave values from a nd b
    if Na == Nb:
        slices = [slice(None,2*Na,2), slice(1,2*Na,2)]
        arys = [a,b]
    elif Na < Nb:
        slices = [slice(None,2*Na,2), slice(1,2*Na,2), slice(2*Na,None)]
        arys = [a, b[:Na], b[Na:]]
    else:
        slices = [slice(None,2*Nb,2), slice(1,2*Nb,2), slice(2*Nb,None)]
        arys = [a[:Nb], b, a[Nb:]]
    
    #put them in
    for slc, ary in zip(slices,arys):
        c[slc,...] = ary
    
    #return things to the original shape
    a, b, c = [np.swapaxes(ary, axis, 0) for ary  in [a,b,c]]
    
    return c

def mids2edges(mids, start='mid', first='adjacent'):
    """
    Reconstructs bin edges given only the midpoints.
    
    Parameters
    ----------
    mids : 1-D array-like
        A 1-D array or list of the midpoints from which bin edges are to be 
        inferred.
    start : {'left'|'right'|'mid'}, optional
        left : start by assuming the spacing between the first two midpts
            is the same as the spacing between the first two bin edges and
            work from there
        right : same as above, but using the last two midpoints and working
            backwords
        mid : put one bin edge at the middle of the center two midpoints and
            work outwards. If there is an odd number of midpoints, use the
            middle midpoint and the one to the left to start.
    first : {float|'adjcacent'|'linear-i'|'linear-x'|function}, optional
        Width of the starting bin to the spacing between the midpoints to extrapolate the
        width of the first or last bin.
        
        The linear options try to extrapolate the width of the start bin by
        assuming the bin widths follow the same linear change as the midpoint
        spacings. linear-i' assumes a linear change with respect to the bin index
        whereas 'linear-x' assumes a linear change with respect to the bin
        value. These can only be used with start set to
        'left' or 'right'. Note that using 'linear' can produce nonsensical output 
        if the spacing between midpoints does not vary linearly.
        
        Alternatively, a function may be provided that fits the midpoint spacings.
        That function should
        return a fit function (just like scipy's interp1d does), such that if
        result = function(xin,yin), then yout = result(xout) is a valid.
    
    Result
    ------
    edges : np.array
        The inferred bin edges.
    
    Could be accelerated with a cython implementation.
    """
    mids = np.array(mids)
    N = len(mids)
    e = np.zeros(N+1)
    if type(first) is not float and first != 'adjacent' and start == 'mid':
        raise ValueError("Start can only be 'mid' if fit == 'none'.")
      
    if type(first) is float:
        if start == 'left': e[0] = mids[0] - first/2.0
        if start == 'right': e[-1] = mids[-1] + first/2.0
    elif first == 'adjacent':
        if start == 'left': e[0] = mids[0] - (mids[1] - mids[0])/2.0
        if start == 'right': e[-1] = mids[-1] + (mids[-1] - mids[-2])/2.0
    else:
        d = mids[1:] - mids[:-1]
        x = midpts(mids)
        if first == 'linear-x':
            c = np.polyfit(x, d, 1)
            fitfun = lambda x: np.polyval(c, x)
        if first == 'linear-i':
            cdi = np.polyfit(np.arange(N-1), d, 1)
            cix = np.polyfit(x, np.arange(N-1), 2)
            def fitfun(x):
                i = np.polyval(cix, x)
                return np.polyval(cdi, i)
        elif callable(first):
            fitfun = first(x,d)
            
        if start == 'left':
            d0 = fitfun(mids[0])
            e[0] = mids[0] - d0/2.0
        if start == 'right':
            d1 = fitfun(mids[-1])
            e[-1] = mids[-1] + d1/2.0
            
    if start == 'left':
        for i in np.arange(0,N): e[i+1] = 2*mids[i] - e[i]
    if start == 'right':
        for i in np.arange(N-1,-1,-1): e[i] = 2*mids[i] - e[i+1]
    if start == 'mid':
        i = N//2
        e[i] = (mids[i-1] + mids[i])/2.0
        for i in np.arange(i-1,-1,-1): e[i] = 2*mids[i] - e[i+1]
        for i in np.arange(i+1,N): e[i+1] = 2*mids[i] - e[i]
    
    if any(e[1:] - e[:-1] <= 0):
        warnings.warn('There are zero or negative length bins in the output. '
                      'Consider using a different fit or start.', RuntimeWarning)
                            
    return e

def block_edges(ary):
    """
    Returns a list of slices each identifying a block of true and false data.
    """
    a = np.insert(ary, [0, len(ary)], [False, False])
    chng = a[1:] - a[:-1]
    beg, = np.nonzero(chng == 1)
    end, = np.nonzero(chng == -1)
    return beg,end

def empty_arrays(N, dtype=float, shape=None):
    arys = [np.array([],dtype) for i in range(N)]
    if shape != None:
        for a in arys: a.shape = shape
    return arys

def inranges(values, ranges, right=False):
    """Determines whether values are in the supplied list of sorted ranges.
    
    ranges can be a nested list of range pairs ([[x00,x01], [x10, x11],
    [x20, x21], ...]), a single list ([x00,x01,x10,x11,x20,x21,...]), or the
    np.array equivalents. Values are in a range if range[0] <= value < range[1]
    unless right=True, then range[0] < value <= range[1].
    
    Returns a boolean array indexing the values that are in the ranges.
    """
    ranges = np.array(ranges)
    if ranges.ndim > 1: ranges = ranges.ravel()
    return (np.digitize(values, ranges, right) % 2 == 1)

def midpts(ary, axis=None):
    """Computes the midpoints between points in a vector.
    
    Output has length len(vec)-1.
    """
    if type(ary) != np.ndarray: ary = np.array(ary)
    if axis == None:
        return (ary[1:] + ary[:-1])/2.0
    else:
        hi = np.split(ary, [1], axis=axis)[1]
        lo = np.split(ary, [-1], axis=axis)[0]
        return (hi+lo)/2.0

def shorten_jumps(vec,maxjump,newjump=None):
    """Finds jumps > maxjump in a vector of increasing values and shortens the
    jump to newjump.
    
    If newjump is not specified it is set to maxjump.
    
    Returns a vector with large jumps shortened to newjump, the values at
    the midpoint of each new (shortened) jump, and the size of the original
    jumps.
    """
    if not newjump: newjump = maxjump
    vec = vec
    jumps = np.concatenate(([0.0],vec[1:] - vec[:-1]))
    jumpindex = np.nonzero(jumps > maxjump)[0]
    jumplen = jumps[jumpindex]
    jumps[jumpindex] = newjump
    vec_new = jumps.cumsum()
    midjump = (vec_new[jumpindex-1] + vec_new[jumpindex])/2.0
    return vec_new, midjump, jumplen
        
def divvy(ary, bins, keyrow=0):
    """
    Divvys up the points in the input vector or array into the indicated 
    bins.
    
    Row keyrow of the array is used to divvy up the array.
    
    Returns a list of arrays, each array containing the points in the
    corresponding bins. Points outside of the bins are discarded. Bins is a
    vector of the bin edges and must be in ascending order.
    
    I tested this against looping throught the bins and removing points from
    the vector that were in the bin at each iteration and this was slightly
    faster.
    """
    ary = np.asarray(ary)
    if ary.ndim == 1: ary = ary.reshape([1,len(ary)])    
    ivec = np.digitize(ary[keyrow,:], bins)
    divvied = [ary[:, ivec == i] for i in np.arange(1,len(bins))]
    
    return divvied        

def chunkogram(vec, chunksize, weights=None, unsorted=False):
    """
    Like chunk_edges, but used when many points have the same value such
    that chunks sometimes must contain different numbers of points.
    
    If enough points have the same value, the bin edges would be identical. 
    This results in infinite values when computing point densities. To
    circumvent that problem, this function extends the right edge of the bin
    to a value between that of the identical points and the next largest point(s).
    
    Because chunks may not be the same size, the function returns the bin
    edges as well as the counts, much like histogram (hence the name).
    
    This will take a lot longer than regular chunk_edges, so don't use it
    if all the values in the vector are unique.
    """
    v = np.sort(vec) if unsorted else vec
    wtd = weights != None
    
    maxlen = len(v)//chunksize + 1
    edges, count = np.zeros(maxlen+1), np.zeros(maxlen)
    edges[0] = (v[0] + v[1])/2.0
    i, iold, j = chunksize+1l, 1l, 0l
    while True: #TODO: use cython to make this fast
        #if the endpoint and following point have the same value
        while (i+1 < len(v)-1) and (v[i-1] == v[i]): i += 1
        if i+1 >= len(v)-1:
            edges[j+1] = (v[-1] + v[-2])/2.0
            count[j] = np.sum(weights[iold:-1]) if wtd else (len(v) - 1 - iold)
            edges, count = edges[:j+2], count[:j+1]
            return np.array(count), np.array(edges)
        else:
            edges[j+1] = (v[i-1] + v[i])/2.0
            count[j] = np.sum(weights[iold:i]) if wtd else (i - iold)
            iold = i
            i += chunksize
            j += 1

def chunk_edges(vec, chunksize, unsorted=False):
    """Determine bin edges that result in an even number of points in each bin.
    
    Assumes the vector is sorted unless specifically told otherwise with the
    unsorted keyword.
    
    The first and last points will be discarded. If you wish to include these
    points, tacking pretend points on both ends of the input vector can
    accomodate the situation.
    """
    v = np.sort(vec) if unsorted else vec
        
    #use indices bc if len(vec) % chunksize == 0 there will be a point left of
    #the last bin edges, but not right of it
    iright = np.arange(1, len(v), chunksize, int)
    
    edges = (v[iright] + v[iright-1])/2.0
    return edges
    
def chunk_sum(vec, chunksize):
    """Computes the sums of chunks of points in a vector.
    """
    Nchunks = len(vec)//chunksize
    end = Nchunks*chunksize
    arr = np.reshape(vec[:end], [Nchunks, chunksize])
    sums = np.sum(arr, 1)
    return sums

def intergolate(x_bin_edges,xin,yin):
    """Compute average of xin,yin within supplied bins.
    
    This funtion is similar to interpolation, but averages the curve repesented
    by xin,yin over the supplied bins to produce the output, yout.
    
    This is particularly useful, for example, for a spectrum of narrow emission
    incident on a detector with broad pixels. Each pixel averages out or 
    "dilutes" the lines that fall within its range. However, simply 
    interpolating at the pixel midpoints is a mistake as these points will
    often land between lines and predict no flux in a pixel where narrow but
    strong lines will actually produce significant flux.
    
    Note that bins outside of xin will assume the curve is constant
    at the value of its closest endpoint.
    """
    
    xbe = np.copy(x_bin_edges)
    
    #if bins extend beyond the range of xin, add extra xin,yin points at the
    #relevant end
    if xbe[0] < xin[0]:
        xin, yin = np.insert(xin,0,xbe[0]), np.insert(yin,0,yin[0])
    if xbe[-1] > xin[-1]:
        xin, yin = np.append(xin,xbe[-1]), np.append(yin,yin[-1])
    
    #define variables to store the slice edges for points within each 
    #intergolation bin, exclusive
    i0, i1 = 0, 0
    yout = np.zeros(len(xbe)-1)
    while xin[i0+1] <= xbe[0]: i0 += 1
    for j in range(len(xbe)-1):
        #find the index of the first point falling just left or on the edge
        #of the bin
        while xin[i1] < xbe[j+1]: i1 += 1
            #TODO: fix the above
        
        #inegrate xin,yin that fall in the current bin
        xedges = np.hstack([[xbe[j]], xin[i0:i1], xbe[j+1]])
        dx = xedges[1:] - xedges[:-1]
        areas = dx*yin[i0:i1]
        yout[j] = np.sum(areas)/(xbe[j+1] - xbe[j])
        
        # update the point just inside of the left bin edge
        i0 = i1
        
    return yout

def rebin_or(newbins, oldbins, values):
    """
    Rebin data using a bitwise or (instead of summing) to determine the values 
    in the new bins.
    
    Does not check that the newbins fall wtihing the bounds of the oldbins. 
    """    
    binmap = np.searchsorted(newbins, oldbins)
    
    
def rebin(newbins, oldbins, values):
    """
    Take binned data and estimate the values wihtin different bins edges.
    
    Assumes the value in each bin is the integral of some function over that
    bin, -NOT- the average of the function.
    """
    binmin, binmax = np.min(oldbins), np.max(oldbins)
    if any(newbins < binmin) or any(newbins > binmax):
        raise ValueError('New bin edges cannot extend beyond old bin edges.')
    
    #clean out non-finite values (so that the cumulative sum isn't screwed up)
    bad = np.logical_not(np.isfinite(values))
    badvals = values[bad]
    values[bad] = 0.0
    
    #generate cumulative integral value at old bin edges, starting at 0.0 for
    #the left edge of the first bin
    integral = np.append(0.0, np.cumsum(values))
    
    #add bad values back in so that nan's and infinites get propagated into
    #any new bin that contains one
    integral[bad] = badvals
    
    #interpolate to the value at the new bin edges
    newintegral = np.interp(newbins, oldbins, integral)
    
    #subtract to get the value of the integral just across each bin
    return np.diff(newintegral)
    
def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]
        
def polyfit_binned(bins, y, yerr, order):
    """Generates a function for the maximum likelihood fit to a set of binned
    data.
    
    Parameters
    ----------
    bins : 2D array-like, shape Nx2
        Bin edges where bins[0] gives the left edges and bins[1] the right.
    y : 1D array-like, length N
        data, the integral of some value over the bins
    yerr : 1D array-like, length N
        errors on the data
    order : int
        the order of the polynomial to fit
    
    Returns
    -------
    coeffs : 1D array
        coefficients of the polynomial, highest power first (such that it
        may be used with numpy.polyval)
    fitfunc : function
        Function that evaluates y and yerr when given a new bin using the 
        maximum likelihood model fit.
    """
    N, M = order, len(y)
    if type(yerr) in [int,float]: yerr = yerr*np.ones(M)
    bins = np.asarray(bins)
    
    #some prelim calcs. all matrices are (N+1)xM
    def prelim(bins, M):
        a, b = bins[:,0], bins[:,1]
        apow = np.array([a**(n+1) for n in range(N+1)])
        bpow = np.array([b**(n+1) for n in range(N+1)])
        bap = bpow - apow
        frac = np.array([np.ones(M)/(n+1) for n in range(N+1)])
        return bap, frac
    
    bap, frac = prelim(bins, M)
    var = np.array([np.array(yerr)**2]*(N+1))
    ymat = np.array([y]*(N+1))
    
    #build the RHS vector
    rhs = np.sum(ymat*bap/var, 1)
    
    #build the LHS coefficient matrix
    nmat = bap/var #N+1xM (n,m)
    kmat = np.transpose(bap)*np.transpose(frac) #MxN+1 (m,k)
    lhs = np.dot(nmat,kmat)
    
    #solve for the polynomial coefficients
    c = np.linalg.solve(lhs, rhs)
    
    #compute the inverse covariance matrix (same as Hessian)
    H = np.dot(nmat*frac,kmat)
    cov = np.linalg.inv(H)
    
    #construct the function to compute model values and errors
    def f(bins):
        M = len(bins)
        bap, frac = prelim(bins, M)
        
        #values
        cmat = np.transpose([c]*M)
        y = np.sum(bap*cmat*frac, 0)
        
        #errors
        T = bap*frac
        yvar = np.dot(np.transpose(T), np.dot(cov, T))
        yerr = np.sqrt(np.diagonal(yvar))
            
        return y, yerr
        
    return c[::-1], cov[::-1,::-1], f
    
def argextrema(y,separate=True):
    """Returns the indices of the local extrema of a series. When consecutive 
    points at an extreme have the same value, the index of the first is
    returned.
    """
    delta = y[1:] - y[:-1]
    pos_neg = np.zeros(len(delta), np.int8)
    pos_neg[delta > 0] = 1
    pos_neg[delta < 0] = -1
    
    curve_sign = pos_neg[1:] - pos_neg[:-1]
    
    if separate:
        argmax = np.nonzero(curve_sign < 0)[0] + 1
        argmin = np.nonzero(curve_sign > 0)[0] + 1
        return argmin,argmax
    else:
        argext = np.nonzero(curve_sign != 0)[0] + 1
        return argext
        
def emd(t,y,Nmodes=None):
    """Decompose function into "intrinsic modes" using empirical mode
    decompisition.
    
    From Huang et al. (1998; RSPA 454:903).
    
    Returns c,r, where c is a list of vectors giving the intrinsic mode values 
    at t and r is a vector giving the residual values at t.
    """
    
    c = []
    h, r = [y]*2
    hold = np.zeros(y.shape)
    while True:
        try:
            while True:
                h = sift(t,h)
                var = np.sum((h-hold)**2/hold**2)
                if var < 0.25:
                    c.append(h)
                    r = r - h
                    
                    #if the user doesn't want any more modes
                    if len(c) == Nmodes:
                        return c,r
                    
                    h = r
                    hold = np.zeros(y.shape)
                    break
                hold = h
        except FlatFunction: #if the residue is has too few extrema
            return c, r
            
class FlatFunction(Exception):
    pass

def sift(t,y,nref=100,plot=False):
    """Identify the dominant "intinsic mode" in a series of data.
    
    See Huang et al. (1998; RSPA 454:903).
    
    Identifies the relative max and min in the series, fits spline curves
    to these to estimate an envelope, then subtracts the mean of the envelope
    from the series. The difference is then returned. The extrema are refelcted
    about the extrema nearest each end of the series to mitigate end
    effects, where nref controls the maximum total number of extrema (max and
    min) that are reflected.
    """
    
    #identify the relative extrema
    argext = argextrema(y, separate=False)
    
    #if there are too few extrema, raise an exception
    if len(argext) < 2:
        raise FlatFunction('Too few max and min in the series to sift')
    
    #should we include the right or left endpoints? (if they are beyond the
    #limits set by the nearest two extrema, then yes)
    inclleft = not inranges(y[[0]], y[argext[:2]])[0]
    inclright = not inranges(y[[-1]], y[argext[-2:]])[0]
    if inclleft and inclright: argext = np.concatenate([[0],argext,[-1]])
    if inclleft and not inclright: argext = np.insert(argext,0,0)
    if not inclleft and inclright: argext = np.append(argext,-1)
    #if neither, do nothing
    
    #now reflect the extrema about both sides
    text, yext  = t[argext], y[argext]
    tleft, yleft = text[0] - (text[nref:0:-1] - text[0]) , yext[nref:0:-1]
    tright, yright = text[-1] + (text[-1] - text[-2:-nref-2:-1]), yext[-2:-nref-2:-1]
    tall = np.concatenate([tleft, text, tright])
    yall = np.concatenate([yleft, yext, yright])
    
    #parse out the min and max. the extrema must alternate, so just figure out
    #whether a min or max comes first
    if yall[0] < yall[1]:
        tmin, tmax, ymin, ymax = tall[::2], tall[1::2], yall[::2], yall[1::2]
    else: 
        tmin, tmax, ymin, ymax = tall[1::2], tall[::2], yall[1::2], yall[::2]
    
    #check again if there are enough extrema, now that the endpoints may have
    #been added
    if len(tmin) < 4 or len(tmax) < 4:
        raise FlatFunction('Too few max and min in the series to sift')
        
    #compute spline enevlopes and mean
    spline_min, spline_max = map(interp1d, [tmin,tmax], [ymin,ymax], ['cubic']*2)
    m = (spline_min(t) + spline_max(t))/2.0
    h = y - m
    
    if plot:
        plt.ioff()
        plt.plot(t,y,'-',t,m,'-')
        plt.plot(tmin,ymin,'g.',tmax,ymax,'k.')
        tmin = np.linspace(tmin[0],tmin[-1],1000)
        tmax = np.linspace(tmax[0],tmax[-1],1000)
        plt.plot(tmin,spline_min(tmin),'-r',tmax,spline_max(tmax),'r-')
        plt.show()
    
    return h