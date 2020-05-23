# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 11:43:52 2014

@author: Parke
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import warnings
from crebin.rebin import rebin as crebin
from .my_numpy_bkwd import *
from astropy import constants as _const, units as _u
from scipy.special import wofz
import mpmath
import inspect
import functools
h, c, k_B = _const.h, _const.c, _const.k_B

#------------------------------------------------------------------------------
# backwards compatability

def rebin_or(newbins, oldbins, oldvalues):
    """Deprecated as of 2015-02-15. Use rebin instead."""
    return rebin(newbins, oldbins, oldvalues, method='or')

def rebin_special(newbins, oldbins, oldvalues, function):
    """Deprecated as of 2015-02-15. Use rebin instead."""
    return rebin(newbins, oldbins, oldvalues, method=function)


#------------------------------------------------------------------------------


def rangeset_intersect(ranges0, ranges1, presorted=False):
    """
    Return the intersection of two sets of sorted ranges, given as Nx2 array-like.
    """

    if len(ranges0) == 0 or len(ranges1) == 0:
        return np.empty([0, 2])
    rng0, rng1 = list(map(np.asarray, [ranges0, ranges1]))
    rng0, rng1 = [np.reshape(a, [-1, 2]) for a in [rng0, rng1]]

    if not presorted:
        rng0, rng1 = [r[np.argsort(r[:,0])] for r in [rng0, rng1]]
    for rng in [rng0, rng1]:
        assert np.all(rng[1:] > rng[:-1])

    l0, r0 = rng0.T
    l1, r1 = rng1.T

    lin0 = inranges(l0, rng1, [1, 0])
    rin0 = inranges(r0, rng1, [0, 1])
    lin1 = inranges(l1, rng0, [0, 0])
    rin1 = inranges(r1, rng0, [0, 0])

    #keep only those edges that are within a good area of the other range
    l = weave(l0[lin0], l1[lin1])
    r = weave(r0[rin0], r1[rin1])
    return np.array([l, r]).T
range_intersect = rangeset_intersect


def rangeset_invert(ranges):
    if len(ranges) == 0:
        return np.array([[-np.inf, np.inf]])
    ranges = np.asarray(ranges)
    edges = ranges.ravel()
    rnglist = [edges[1:-1].reshape([-1, 2])]
    if edges[0] != -np.inf:
        firstrng = [[-np.inf, edges[0]]]
        rnglist.insert(0, firstrng)
    if edges[-1] != np.inf:
        lastrng = [[edges[-1], np.inf]]
        rnglist.append(lastrng)
    return np.vstack(rnglist)


def rangeset_union(ranges0, ranges1):
    invrng0, invrng1 = list(map(rangeset_invert, [ranges0, ranges1]))
    xinv = range_intersect(invrng0, invrng1)
    return rangeset_invert(xinv)


def rangeset_subtract(baseranges, subranges):
    """Subtract subranges from baseranges, given as Nx2 arrays."""
    return range_intersect(baseranges, rangeset_invert(subranges))


def weave(a, b):
    """
    Insert values from b into a in a way that maintains their order. Both must
    be sorted.
    """
    mapba = np.searchsorted(a, b)
    return np.insert(a, mapba, b)

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
    bktdata = list(map(idata, bkti))

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
    args[0] = np.asarray(args[0])**2
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
    a, b = list(map(np.asarray, [a,b]))

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

def mids2edges(mids, start='mid', first='adjacent', simple=False):
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

    if simple:
        edges = midpts(mids)
        d0 = edges[0] - mids[0]
        d1 = mids[-1] - edges[-1]
        return np.insert(edges, [0, len(edges)], [mids[0] - d0, mids[-1] + d1])

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

def splitsum(ary, indices):
    """
    Splits an array and sums values in each section.

    Parameters
    ----------
    ary : 1-D array-like
        Array to be divided into sub-arrays.
    indices : 1-D array-like
        Integers giving the slice indices for splitting the array.

    Returns
    -------
    sums : 1-D array
        An array containg the sum of each section of the split array. len(sums)
        == len(indices) + 1
    """
    if ary.ndim > 1:
        raise NotImplementedError("Can only hande 1-D arrays at the moment "
        "because I'm not sure how to generalize this to an n-dim case.")

    #add begging and end to the indices and make sure none are negative to
    #avoid wraparound
    indices = np.insert(indices, [0, len(indices)], [0, len(ary)])
    neg = (indices < 0)
    indices[neg] = len(ary) + indices[neg]

    #beginning and end of each block
    begs = indices[:-1]
    ends = indices[1:]

    #cumulative sums, starting with zero
    cs = np.insert(np.cumsum(ary), 0, 0.0)

    return cs[ends] - cs[begs]


def block_edges(ary):
    """
    Returns the beginning and end slice index of each block of true values.
    """
    a = np.insert(ary, [0, len(ary)], [False, False])
    a = a.astype('i1')
    chng = a[1:] - a[:-1]
    beg, = np.nonzero(chng == 1)
    end, = np.nonzero(chng == -1)
    return beg, end


def empty_arrays(N, dtype=float, shape=None):
    arys = [np.array([],dtype) for i in range(N)]
    if shape != None:
        for a in arys: a.shape = shape
    return arys


def smooth(x, n, safe=True):
    """
    Compute an n-point moving average of the data in vector x. Result will have a length of len(x) - (n-1). Using
    save avoids the arithmetic overflow and accumulated errors that can result from using numpy.cumsum, though cumsum
    is (probably) faster.
    """
    assert x.ndim == 1
    if safe:
        m = len(x)
        result = np.zeros(m - (n-1))
        for i in range(n):
            result += x[i:(m-n+i+1)]
        return result / float(n)
    else:
        s = np.cumsum(x)
        s = np.insert(s, 0, 0.0)
        return (s[n:] - s[:-n]) / float(n)


def inranges(values, ranges, inclusive=(False, True)):
    """Determines whether values are in the supplied list of sorted ranges.

    Parameters
    ----------
    values : 1-D array-like
        The values to be checked.
    ranges : 1-D or 2-D array-like
        The ranges used to check whether values are in or out.
        If 2-D, ranges should have dimensions Nx2, where N is the number of
        ranges. If 1-D, it should have length 2N. A 2xN array may be used, but
        note that it will be assumed to be Nx2 if N == 2.
    inclusive : length 2 list of booleans
        Whether to treat bounds as inclusive. Because it is the default
        behavior of numpy.searchsorted, [False, True] is the default here as
        well. Using [False, False] or [True, True] will require roughly triple
        computation time.

    Returns a boolean array indexing the values that are in the ranges.
    """
    inclusive = tuple(inclusive)
    ranges = np.asarray(ranges)
    if ranges.ndim == 2:
        if ranges.shape[1] != 2:
            ranges = ranges.T
        ranges = ranges.ravel()

    if inclusive == (0, 1):
        return (np.searchsorted(ranges, values) % 2 == 1)
    if inclusive == (1, 0):
        return (np.searchsorted(ranges, values, side='right') % 2 == 1)
    if inclusive == (1, 1):
        a = (np.searchsorted(ranges, values) % 2 == 1)
        b = (np.searchsorted(ranges, values, side='right') % 2 == 1)
        return (a | b)
    if inclusive == (0, 0):
        a = (np.searchsorted(ranges, values) % 2 == 1)
        b = (np.searchsorted(ranges, values, side='right') % 2 == 1)
        return (a & b)


def binoverlap(binsa, binsb, method='tight'):
    """
    Returns the boolean indices of the bins in b tha overlap bins in a.

    Parameters
    ----------
    binsa : 1d array-like
        Bin edges. Must be sorted.
    binsb : 1d array-like
        Edges of the bins to check for overlap with binsa.
    method : {'tight'|'loose'}, optional
        tight (default) : exclude bins that only partially overlap the range
            of the other set of bins
        loose : include such bins

    Returns
    -------
    overlap : 1d boolean arrays
        Boolean array where True values indicate overlapping bins.
        len(overlap) == len(binsb) - 1
    """
    binsa, binsb = list(map(np.asarray, [binsa, binsb]))
    rng = binsa[[0,-1]]
    if binsb[0] >= rng[1] or binsb[-1] <= rng[0]:
        return np.zeros(len(binsb) - 1, bool)

    left, right = binsb[:-1], binsb[1:]

    if method == 'tight':
        lin, rin = list(map(inranges, [left, right], [rng]*2, [[1,1]]*2))
        return lin & rin

    if method == 'loose':
        lin, rin = list(map(inranges, [left, right], [rng]*2, [[0,0]]*2))
        result = lin | rin
        #if all of binsa is contained within a single bin in b
        ib = np.searchsorted(binsb, rng) - 1
        if ib[0] == ib[1]:
            result[ib[0]] = True
        return result


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


def shorten_jumps(x, maxjump, newjump=None, ignore_nans=True):
    """Finds jumps > maxjump in a vector of increasing values and shortens the
    jump to newjump.

    If newjump is not specified it is set to maxjump.

    Returns a vector with large jumps shortened to newjump, the values at
    the midpoint of each new (shortened) jump, and the size of the original
    jumps.
    """
    if not newjump:
        newjump = maxjump
    if ignore_nans:
        nans = np.isnan(x)
        if np.any(nans):
            i_nan, = np.nonzero(np.isnan(x))
            x = np.delete(x, i_nan)
    jumps = np.concatenate(([0.0], x[1:] - x[:-1]))
    jumpindex = np.nonzero(jumps > maxjump)[0]
    jumplen = jumps[jumpindex]
    jumps[jumpindex] = newjump
    x_new = jumps.cumsum() + x[0]
    midjump = (x_new[jumpindex-1] + x_new[jumpindex])/2.0
    if ignore_nans and np.any(nans):
        x_new = np.insert(x_new, i_nan - list(range(len(i_nan))), np.nan)
    return x_new, midjump, jumplen


def fold(x, x_fold, dx_fold):
    i_lim = np.searchsorted(x, x_fold)
    i = np.arange(len(x))
    add_bool = i[None, :] >= i_lim[:, None]
    dx_vec = add_bool * dx_fold[:, None]
    dx_vec = np.sum(dx_vec, 0)
    return x - dx_vec


def last_before(x, xlim):
    before = x < xlim
    if not np.any(before):
        return None
    return np.max(x[before])


def arg_last_before(x, xlim):
    before = x < xlim
    if not np.any(before):
        return None
    imx = np.argmax(x[before])
    i, = np.nonzero(before)
    return i[imx]


def first_after(x, xlim):
    after = x > xlim
    if not np.any(after):
        return None
    return np.min(x[after])


def arg_first_after(x, xlim):
    after = x > xlim
    if not np.any(after):
        return None
    imn = np.argmin(x[after])
    i, = np.nonzero(after)
    return i[imn]


def divvy(ary, bins, keyrow=0):
    """
    Divvys up the points in the input vector or array into the indicated
    bins. Points outside the bins are discarded.

    Parameters
    ----------
    ary : 2D array-like
        The array of values, each row a different dimension of the data.
    bins : 1D array-like
        Edges of the bins, similar to histogram. Must be in ascending order.
    keyrow : int
        The row to use for checking if each column is a data point in bin.

    Returns
    -------
    divvied_ary : list
        A list of 2D arrays of length len(bins) - 1.

    """

    """
    I tested this against looping throught the bins and removing points from
    the vector that were in the bin at each iteration and this was slightly
    faster.
    """
    ary = np.asarray(ary)
    if ary.ndim == 1: ary = ary.reshape([1,len(ary)])
    ivec = np.searchsorted(bins, ary[keyrow,:])
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
    i, iold, j = chunksize+1, 1, 0
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

def intergolate(x_bin_edges,xin,yin, left=None, right=None):
    """Compute average of xin,yin within supplied bins.

    This funtion is similar to interpolation, but averages the curve repesented
    by xin,yin over the supplied bins to produce the output, yout.

    This is particularly useful, for example, for a spectrum of narrow emission
    incident on a detector with broad pixels. Each pixel averages out or
    "dilutes" the lines that fall within its range. However, simply
    interpolating at the pixel midpoints is a mistake as these points will
    often land between lines and predict no flux in a pixel where narrow but
    strong lines will actually produce significant flux.

    left and right have the same definition as in np.interp
    """

    x = np.hstack((x_bin_edges, xin))
    x = np.sort(x)
    y = np.interp(x, xin, yin, left, right)
    I = cumtrapz(y, x, True)
    Iedges = np.interp(x_bin_edges, x, I)
    y_bin_avg = np.diff(Iedges)/np.diff(x_bin_edges)
    return y_bin_avg


def cumtrapz(y, x, zero_start=False):
    result = np.cumsum(midpts(y)*np.diff(x))
    if zero_start:
        result = np.insert(result, 0, 0)
    return result


def rebin(newbins, oldbins, values, method='sum'):
    """
    Take binned data and rebin it using the specified method.

    Parameters
    ----------
    newbins : 1d array-like
        The edges of the new bins in ascending order.
    oldbins : 1d array-like
        The edges of the old bins in ascending order.
    values : 1d array-like
        Data values within each bin, len(values) == len(oldbins) - 1.
    method : str or function, optional
        How the data should be rebinned. Several common methods can be
        specified with string input
            - 'sum' : The default. Bin values are summed. Where newbin edges
                fall between oldbin edges, values in those bins are fractioned
                appropriately.
            - 'avg' : Bin values are averaged, weighted by bin widths.
                (Essentially, values are multiplied by the oldbin widths,
                rebinned in a summation sense, then divided by the newbin
                widths.)
            - 'min' : The minumum value of all oldbins all or partly within a
                newbin is taken.
            - 'max' : Same as min, but the maximum value is taken.
            - 'or' : Same as min, but the btwise or of the values is taken.
        A custom function can also be provided as input. This must accept an
        array as input and return a scalar, such as numpy.min.

    Returns
    -------
    newvalues : 1d array
        Rebinned data, len(newvalues) == len(newbins) - 1.
    """
    # vet input
    nb, ob, ov = list(map(np.asarray, [newbins, oldbins, values], [float, float, None]))
    dt = ov.dtype
    assert np.all(oldbins[1:] > oldbins[:-1])
    assert np.all(newbins[1:] > newbins[:-1])
    assert len(oldbins) == len(values) + 1
    binmin, binmax = oldbins[0], oldbins[-1]
    if np.any(nb < binmin) or np.any(nb > binmax):
        raise ValueError('New bin edges cannot extend beyond old bin edges.')

    # use built-in cython function I made for the easy ones
    if method in ['sum', 'min', 'max', 'or']:
        nv = crebin(nb, ob, ov, method)
    elif method == 'avg':
        do, dn = list(map(np.diff, [ob, nb]))
        nv = crebin(nb, ob, ov*do, 'sum') / dn

    # otherwise, split up the array and apply the provided function
    elif callable(method):
        i = np.searchsorted(oldbins, newbins, side='right') #where new bin edges fit into old

        #this is a complicated trick... basically insert a duplicate value into
        #the oldvalues array wherever a newbin edge falls between oldbin edges
        #then add to the i values so that they fall in between the duplicated
        #values. Then split up the array
        noncoincident = (oldbins[i-1] != newbins)
        inc = i[noncoincident]
        expanded = np.insert(values, inc, values[inc-1])
        ie = i + np.cumsum(noncoincident) - 1

        nv = np.array(list(map(method, np.split(expanded, ie)[1:-1])))
    else:
        raise ValueError('Method not recognized.')

    assert len(nv) == len(nb) - 1
    return nv.astype(dt)


def interp_roots(x, y):
    """
    Find the roots of some data by linear interpolation where the y values cross the x-axis.

    For series of zeros, the midpoint of the zero values is given.

    Parameters
    ----------
    x : array
    y : array

    Returns
    -------
    x0 : array
        x values of the roots
    """
    sign = np.sign(y)

    # zero values where data crosses x axis
    c = crossings = np.nonzero(abs(sign[1:] - sign[:-1]) == 2)[0] + 1
    a = np.abs(y)
    x0_crossings = (x[c]*a[c-1] + x[c-1]*a[c])/(a[c] + a[c-1])

    # zero values where data falls on x axis
    zero_start, zero_end = block_edges(y == 0)
    x0_zero = (x[zero_start] + x[zero_end-1]) / 2.0

    return np.unique(np.hstack([x0_crossings, x0_zero]))


def corrcoef(x, y):
    """
    Compute the correlation coefficient between rows of x and y.
    """
    if x.ndim == 1:
        x = np.reshape(x, [1, len(x)])
        y = np.reshape(y, [1, len(y)])
    mx, my = [np.mean(a, 1) for a in [x, y]]
    x0, y0 = x - mx[:,np.newaxis], y - my[:,np.newaxis]
    Sxy = np.sum(x0*y0, 1)
    Sxx = np.sum(x0*x0, 1)
    Syy = np.sum(y0*y0, 1)
    return Sxy/(np.sqrt(Sxx)*np.sqrt(Syy))


def flagruns(x):
    """
    Flag portions of x where succesive points have the same value.
    """
    x = np.asarray(x)
    leqr = (x[:-1] == x[1:])
    leqr = np.insert(leqr, [0, len(leqr)], [False, False])
    flags = (leqr[:-1] | leqr[1:])
    return flags


def runslices(x,endpts=False):
    """
    Return the slice indices that separate runs in x. Great for use with splitsum.
    """
    # first all indices where value crosses from positive to negative or vice versa
    pospts = (x > 0)
    negpts = (x < 0)
    arg_cross = np.nonzero((pospts[:-1] & negpts[1:]) | (negpts[:-1] & pospts[1:]))[0] + 1

    # now all indices of the middle zero in all series of zeros
    zeropts = (x == 0)
    zero_beg, zero_end = block_edges(zeropts)
    arg_zero = (zero_beg + zero_end + 1) // 2

    insert_zeros_at = np.searchsorted(arg_cross, arg_zero)
    splits = np.insert(arg_cross, insert_zeros_at, arg_zero)
    if endpts:
        return np.insert(splits, [0, len(splits), [0, len(x)]])
    else:
        return splits


def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
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
    covar : 2D array
        covariance matrix for the polynomial coefficients
    fitfunc : function
        Function that evaluates y and yerr when given a new bin using the
        maximum likelihood model fit.
    """
    N, M = order, len(y)
    if type(yerr) in [int,float]: yerr = yerr*np.ones(M)
    bins = np.asarray(bins)
    assert not np.any(yerr == 0.0)

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
    kmat = bap.T*frac.T #MxN+1 (m,k)
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
        cT = np.dot(cov, T)
        #to avoid memory overflow, compute diagonal elements directly instead
        #of dotting the matrices
        yvar = np.sum(T*cT, 0)
        yerr = np.sqrt(yvar)

        return y, yerr

    return c[::-1], cov[::-1,::-1], f


def chi2normseries(x, xerr, y, yerr):
    """
    Find the normalization factor, c, to apply to y that minimizes the
    chi-square statistic between x and c*y.

    The errors in y are also assumed to scale with c.
    """
    #there is a word file with derivation of this math
    vx, vy = xerr**2, yerr**2

    #assuming the errors don't scale...
#    A = np.nansum(x*y/vx) + np.nansum(x*y/vy)
#    B = np.nansum(y*y/vx) + np.nansum(y*y/vy)
#    return A/B

    #assuming the errors scale
    Sxyx = np.nansum(x*y/vx)
    Syyx = np.nansum(y*y/vx)
    Sxyy = np.nansum(x*y/vy)
    Sxxy = np.nansum(x*x/vy)
    r = np.roots([Syyx, -Sxyx, 0.0, Sxyy, -Sxxy])
    #select the root with a positive real part and the smallest possible
    #imaginary part (would be zero except for arithmetic error)
    r = r[np.real(r) > 0]
    r = r[np.argmin(np.abs(np.imag(r)))]
    return float(np.real(r))


def voigt_xsection(w, w0, f, gamma, T, mass, b=None):
    """
    Compute the absorption cross section using hte voigt profile for a line.

    Parameters
    ----------
    w : astropy quantity array or scalar
        Scalar or vector of wavelengths at which to compute cross section.
    w0: quanitity
        Line center wavelength.
    f: scalar
        Oscillator strength.
    gamma: quantity
        Sum of transition rates (A values) out of upper and lower states. Just Aul for a resonance line where only
        spontaneous decay is an issue.
    T: quantity
        Temperature of the gas. Can be None if you provide a b value instead.
    mass: quantity
        molecular mass of the gas
    b : quantity
        Doppler b value (in velocity units) of the line
    Returns
    -------
    x : quantity
        Cross section of the line at each w.
    """

    nu = _const.c / w
    nu0 = _const.c / w0
    if T is None:
        sigma_dopp = b/_const.c*nu0/np.sqrt(2)
    else:
        sigma_dopp = np.sqrt(_const.k_B*T/mass/_const.c**2) * nu0
    dnu = nu - nu0
    gauss_sigma = sigma_dopp.to(_u.Hz).value
    lorentz_FWHM = (gamma/2/np.pi).to(_u.Hz).value
    phi = voigt(dnu.to(_u.Hz).value, gauss_sigma, lorentz_FWHM) * _u.s
    x = np.pi*_const.e.esu**2/_const.m_e/_const.c * f * phi
    return x.to('cm2')


def voigt_emission(w, w0, gamma, b):
    """
    Compute a voigt emission profile, normalized so that it will integrate to unity.

    Parameters
    ----------
    w : astropy quantity array or scalar
        Scalar or vector of wavelengths at which to compute cross section.
    w0: quanitity
        Line center wavelength.
    gamma: quantity
        Sum of transition rates (A values) out of upper and lower states. Just Aul for a resonance line where only
        spontaneous decay is an issue.
    b : quantity
        Doppler b value (in velocity units) of the line
    Returns
    -------
    x : quantity
        Cross section of the line at each w.
    """

    nu = _const.c / w
    nu0 = _const.c / w0
    sigma_dopp = b/_const.c*nu0/np.sqrt(2)
    dnu = nu - nu0
    gauss_sigma = sigma_dopp.to(_u.Hz).value
    lorentz_FWHM = (gamma/2/np.pi).to(_u.Hz).value
    phi_nu = voigt(dnu.to(_u.Hz).value, gauss_sigma, lorentz_FWHM) * _u.s
    phi_lam = phi_nu * (c/w**2)
    return phi_lam.to(w.unit**-1)


def voigt(x, gauss_sigma, lorentz_FWHM):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    Line function is normalized to integrate to unity.

    """
    #FIXME replace with astropy version once version 1.1.2 is out

    sigma = gauss_sigma
    gamma = lorentz_FWHM/2.0
    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi)


def align(a, b):
    """
    Align two 1D arrays by maximizing the cross correlation.

    Parameters
    ----------
    a
    b

    Returns
    -------
    offset
        integer number of elements which b must be translated to maximize correlation with a, referenced from simply
        starting both at index of 0 and cutting off the shorter.

    Examples
    --------
    a = np.array([0, 0, 1, 0])
    b = np.array([1, 0, 0, 0, 0])
    offset = mnp.align(a,b)

    """

    # assume b is shorter than a
    if len(b) <= len(a):
        corr = np.correlate(a, b, 'full')
        return np.argmax(corr) - len(b) + 1
    else:
        return -align(b, a)


_Li = mpmath.fp.polylog
def _P3(x):
    e = np.exp(-x)
    return _Li(4, e) + x*_Li(3, e) + x**2/2*_Li(2, e) + x**3/6*_Li(1, e)
_P3 = np.vectorize(_P3)


def blackbody_integral_cumulative(T, w):
    """Integral of Planck function from 0 to w, computed analytically so it is fast. No extra factor of pi needed,
    the integral from 0 to inf would equal stefan-boltzmann law. However, 0 and inf are not acceptable input."""
    x = (h*c/w/k_B/T).to('').value
    I = 12 * np.pi * (k_B*T)**4 / c**2 / h**3 * _P3(x)
    return I.to('erg s-1 cm-2')
planck_integral_cumulative = blackbody_integral_cumulative

def blackbody(T, w):
    "Planck function that will integrate to the same value as stefan-boltzmann. "
    C = 2 * np.pi * _const.h * _const.c ** 2
    eC = _const.h * _const.c / _const.k_B
    exponent = (eC / T / w).to('').value
    return (C / w ** 5 / (np.exp(exponent) - 1)).to('erg s-1 cm-2 AA-1')
planck = blackbody


def blackbody_integral(T, w0, w1):
    """Integral of Planck function from w0 to  w1, computed analytically so it is fast. No extra factor of pi needed,
    the integral from 0 to inf would equal stefan-boltzmann law. However, 0 and inf are not acceptable input."""
    if w0 == 0:
        return blackbody_integral_cumulative(T, w1)
    w = [w0.value, w1.to(w0.unit).value]*w0.unit
    return np.diff(blackbody_integral_cumulative(T, w))[0]
planck_integral = blackbody_integral


def ratio_err(num, enum, denom, edenom):
    return num/denom * np.sqrt((enum/num)**2 + (edenom/denom)**2)


def doppler_shift(w, velocity):
    return (1 + velocity/_const.c)*w


def w2v(w, w0):
    v = (w - w0)/w0 * _const.c
    return v.to('km/s')


def scalar_or_array(*names):
    """
    Decorator to make a function take either scalar or array input and return
    either scalar or array accordingly.

    The decorator accepts as arguments the names of all the parameters that
    should  be turned into arrays if the user provides scalars. Names should
    be strings.

    The function must be able to handle array input for all of the named
    arguments.

    In  operation, if all the  named arguments are scalars, then the
    decorator will apply np.squeeze() to everything the function returns.


    Example:
    @mnp.scalar_or_array('x', 'y')
    def test(x, y):
    x[x > 10] = 0
        return x, y

    test(5, 0)
    # returns: (array(5), array(0))

    test(20, 0)
    # returns: (array(0), array(0))

    test(np.arange(20), 0)
    # returns: (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  0,  0,  0,
                        0,  0,  0,  0,  0,  0]), array([0]))
    # notice that the second value returned gets turned into an array in
    # this case
    """
    def decorator(func):

        # get the decorated functions call signature
        signature = inspect.signature(func)

        # now the modified version of the decorated function
        @functools.wraps(func)
        def mod(*args, **kwargs):
            # map this modified function's arguments to that of the decorated
            # function through the "bind" method of the signature
            boundargs = signature.bind(*args, **kwargs)

            # now check if each of the listed arguments is a scalar. if so,
            # make it an array with ndim=1 in the bound arguments.
            scalar_input = []
            for name in names:
                if name in signature.parameters:
                    val = boundargs.arguments[name]
                    if np.isscalar(val) or \
                            (hasattr(val, 'isscalar') and val.isscalar):
                        scalar_input.append(True)
                        ary = np.reshape(val, 1)
                        boundargs.arguments[name] = ary
                    else:
                        scalar_input.append(False)

            # now apply the function
            result = func(**boundargs.arguments)

            # if all the user-named inputs were scalars, then try to return
            # all scalars, else, just return what the functon spit  out
            if all(scalar_input):
                if type(result) is tuple:
                    return tuple(map(np.squeeze, result))
                else:
                    return np.squeeze(result)
            else:
                return result

        return mod
    return decorator


def polyerr(p, x, C):
    """compute errors for polynomial values at x given a covariance matrix for the polynomial fit
    This factors in the errors on the polynomial fit (and their correlation) when predicting values of the polynomial at x.

    Example:
    xpts = np.arange(10) + 0.5*np.random.randn(10)
    ypts = 2*np.arange(10) + np.random.randn(10)
    plt.plot(xpts, ypts, '.')
    p, C = np.polyfit(xpts,ypts,1,cov=1)
    x = np.linspace(-1, 10, 100)
    y = np.polyval(p, x)
    yerr = mnp.polyerr(p, x, C)
    plt.fill_between(x, y-yerr, y+yerr)
    """

    # compute a list of transformation matrices -- one for each x
    # these are the partial(y)/partial(p[i]) for each i of the polynomial
    # i.e. you are taking the partial derivs with respect to the parameters

    powers = np.arange(len(p))[::-1]
    partials = x[:,None]**powers[None,:]

    prod1 = np.inner(partials, C)
    prod2 = np.sum(prod1*partials, 1)

    return np.sqrt(prod2)

