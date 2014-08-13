# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 11:43:52 2014

@author: Parke
"""
import numpy as np
from scipy.interpolate import interp1d #InterpolatedUnivariateSpline as ius #pchip_interpolate
import pdb
import matplotlib.pyplot as plt

def midpts(ary, axis=None):
    """Computes the midpoints between points in a vector.
    
    Output has length len(vec)-1.
    """
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
    """Divvys up the points in the input vector or array into the indicated 
    bins.
    
    Row keyrow of the array is used to divvy up the array.
    
    Returns a list of arrays, each array containing the points in the
    corresponding bins. Points outside of the bins are discarded. Bins is a
    vector of the bin edges and must be in ascending order.
    
    I tested this against looping throught the bins and removing points from
    the vector that were in the bin at each iteration and this was slightly
    faster.
    """
    list_in, ashape = (type(ary) == list), ary.shape
    if list_in: ary = np.array(ary)
    if ary.ndim == 1: ary.resize([1,len(ary)])    
    ivec = np.digitize(ary[keyrow,:], bins)
    divvied = [ary[:, ivec == i] for i in np.arange(1,len(bins))]
    
    #return ary to it's original form
    if list_in: ary = list(ary)
    if len(ashape) == 1: ary.resize(ashape)
    
    return divvied        

def chunkogram(vec, chunksize, weights=None, unsorted=False):
    """Like chunk_edges, but used when many points have the same value such
    that chunks sometimes must contain different numbers of points.
    
    If enough points have the same value, the bin edges would be identical. 
    This results in infinite values when computing point densities. To
    circumvent that problem, this function extends the right edge of the bin
    to a value between that of the identical points and the next largest point.
    
    Because chunks may not be the same size, the function returns the bin
    edges as well as the counts, much like histogram (hence the name).
    
    This will take a lot longer than regular chunk_edges, so don't use it
    if all the values in the vector are unique.
    """
    v = np.sort(vec) if unsorted else v = vec
    wtd = weights != None
    
    edges, count = [(v[0] + v[1])/2.0], []
    i, iold = chunksize, 0
    while i+1 < len(v)-1:
        #if the endpoint and following point have the same value
        while v[i] == v[i+1]:
            i += 1
            if i+1 >= len(v)-1:
                edges.append(v[-1])
                count.append(np.sum(weights[iold:i]) if wtd else (i - iold))
                break
        edges.append((v[i] + v[i+1])/2.0)
        count.append(np.sum(weights[iold:i]) if wtd else (i - iold))
        iold = i
        i += chunksize
        
    return np.array(count), np.array(edges)
        

def chunk_edges(vec, chunksize, unsorted=False):
    """Determine bin edges that result in an even number of points in each bin.
    
    Assumes the vector is sorted unless specifically told otherwise with the
    unsorted keyword.
    
    The first and last points will be discarded. If you wish to include these
    points, tacking pretend points on both ends of the input vector can
    accomodate the situation.
    """
    v = np.sort(vec) if unsorted else v = vec
        
    #use indices bc if len(vec) % chunksize == 0 there will be a point left of
    #the last bin edges, but not right of it
    iright = np.arange(1, len(v), chunksize)
    
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
    from numpy import concatenate, trapz, insert, append, zeros
    #import pdb
    
    xbe = x_bin_edges    
    
    #if bins extend beyond the range of xin, add extra xin,yin points at the
    #relevant end
    if xbe[0] < xin[0]:
        xin, yin = insert(xin,0,xbe[0]), insert(yin,0,yin[0])
    if xbe[-1] > xin[-1]:
        xin, yin = append(xin,xbe[-1]), append(yin,yin[-1])
    
    #make a single-point interpolations function, mainly for readability
    interp = lambda x0,y0,x1,y1,x: (y1-y0)/(x1-x0)*(x-x0) + y0    
    
    #define variables to store the indices of the points just right of the left
    #and right edges of the bins, respectively
    i0, i1 = 0, 0
    yout = zeros(len(xbe)-1)
    while xin[i0] < xbe[0]: i0 += 1
    for j in range(len(xbe)-1):
        #find the index of the first point falling just right or on the edge
        #of the bin
        while xin[i1] < xbe[j+1]: i1 += 1
        
        #interpolate values at the bin edges
        yleft = interp(xin[i0-1], yin[i0-1], xin[i0], yin[i0], xbe[j])
        yright = interp(xin[i1-1], yin[i1-1], xin[i1], yin[i1], xbe[j+1])
        
        #inegrate xin,yin that fall in the current bin
        xsnippet = concatenate(([xbe[j]],xin[i0:i1],[xbe[j+1]]))
        ysnippet = concatenate(([yleft],yin[i0:i1],[yright]))
#        pdb.set_trace()
        yout[j] = trapz(x = xsnippet, y = ysnippet)/(xbe[j+1] - xbe[j])
        
        # update the point just inside of the left bin edge
        i0 = i1
        
    return yout

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]
        
def polyfit_binned(bins, y, yerr, order):
    """Generates a function for the maximum likelihood fit to a set of binned
    data.
    
    bins = [[a0, b0], ..., [aM, bM]] are the bin edges
    y = data, the integral of some value over the bins
    
    return the coefficents of the polynomial ([c0, c1, .. cN]) where N=order
    the covariance matrix (!! not sigma, but sigma**2), and a function that 
    evaluates y and yerr when given a new bin using the maximum likelihood 
    model fit
    """
    N, M = order, len(y)
    if type(yerr) in [int,float]: yerr = yerr*np.ones(M)
    
    #some prelim calcs. all matrices are (N+1)xM
    def prelim(bins, M):
        bins = np.array(bins)
        a, b = bins[:,0], bins[:,1.]
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
        
    return c, cov, f
    
def argextrema(y):
    """Returns the indices of the local extrema of a series. When consecutive 
    points at an extreme have the same value, the index of the first is
    returned.
    """
    delta = y[1:] - y[:-1]
    pos_neg = delta//abs(delta)
    curve_sign = pos_neg[1:] - pos_neg[:-1]
    argmax = np.nonzero(curve_sign < 0)[0] + 1
    argmin = np.nonzero(curve_sign > 0)[0] + 1
    return argmin,argmax
    
def emd(t,y,Nmodes=None):
    """Decompose function into "intrinsic modes" using empirical mode
    decompisition.
    
    From Huang et al. (1998; RSPA 454:903)
    """
    c = []
    h, r = [y]*2
    hold = np.zeros(y.shape)
    while True:
        try:
            while True:
                h = sift5(t,h)
                var = np.sum((h-hold)**2/hold**2)
                if var < 0.25:
                    c.append(h)
                    r = r - h
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

def sift4(t,y):
    #identify the relative extrema
    argmin, argmax = argextrema(y)
    
    #if there are too few extrema, raise an exception
    n = 2
    if len(argmin) < n or len(argmax) < n:
        raise FlatFunction('Fewer than {} max or min in the series -- cannot sift.'.format(n))
    
    #function to mirror nearest two extrema about the end
    nref = 4
    def reflect(i):
        s = -1 if i < 0 else 1
        #parse out the end and extrema points
        tend, yend = t[i], y[i]
        [tmin,ymin],[tmax,ymax] = [[t[ii:ii+nref*s:s],y[ii:ii+nref*s:s]] for ii in 
                                   [argmin[i],argmax[i]]]
        
        #mirror the points about the end if it is outside of the two extrema
        #else mirror about the extremum nearest the end
        if yend > ymax[0] or yend < ymin[0]: #if the end is outside the relative extrema
            taxis = tend
            if yend > ymax[0]: 
                tmax = np.insert(tmax,0,tend)
                ymax = np.insert(ymax,0,yend)
            if yend < ymin[0]:
                tmin = np.insert(tmin,0,tend)
                ymin = np.insert(ymin,0,yend)
        else:
            rng = lambda v: (v[i+s], v[i+s*(nref+1)])
            if abs(tmin[0]-tend) < abs(tmax[0]-tend): #if min is closest to the end
                taxis = tmin[0]
                i0,i1 = rng(argmin)
                tmin,ymin = t[i0:i1:s], y[i0:i1:s]
            else:
                taxis = tmax[0]
                i0,i1 = rng(argmax)
                tmax,ymax = t[i0:i1:s], y[i0:i1:s]
        
        if s == 1:
            tmin, tmax, ymin, ymax = [x[::-1] for x in [tmin, tmax, ymin, ymax]]
        tmirrored = [taxis + (taxis - tmin), taxis + (taxis - tmax)]
        ymirrored = [ymin,ymax]
        return tmirrored, ymirrored
    
    #get the mirrored times and construct the vectors with extended ends
    [tleft, yleft], [tright, yright] = map(reflect, [0,-1])
    tmin, tmax, ymin, ymax = map(np.concatenate, ([tleft[0], t[argmin], tright[0]],
                                                  [tleft[1], t[argmax], tright[1]],
                                                  [yleft[0], y[argmin], yright[0]],
                                                  [yleft[1], y[argmax], yright[1]]))
    
    #compute spline enevlopes and mean
    spline_min, spline_max = map(interp1d, [tmin,tmax], [ymin,ymax], ['cubic']*2)
    m = (spline_min(t) + spline_max(t))/2.0
    h = y - m
    
    return h

def sift5(t,y):
    #identify the relative extrema
    argmin, argmax = argextrema(y)
    
    #if there are too few extrema, raise an exception
    n = 2
    if len(argmin) < n or len(argmax) < n:
        raise FlatFunction('Fewer than {} max or min in the series -- cannot sift.'.format(n))
    
    nx = 100
    def extend(tt,yy):
        reflect = lambda v,v0: v0 + (v0 - v)
        tleft, yleft = [x[nx:0:-1] for x in [tt,yy]]
        tright, yright = [x[-1:-1-nx:-1] for x in [tt,yy]]
        tleft, tright = map(reflect, [tleft, tright], [t[0], t[-1]])
        tout = np.concatenate([tleft, tt, tright])
        yout = np.concatenate([yleft, yy, yright])
        return tout, yout
    
    #construct vectors with extended ends
    [tmin,ymin], [tmax,ymax] = map(extend, [t[argmin],t[argmax]], 
                                           [y[argmin],y[argmax]])
    
    #insert ends as necessary
    blah = lambda tt,yy: [np.insert(x,nx,v[0]) for x,v in [[tt,t], [yy,y]]]
    blah2 = lambda tt,yy: [np.insert(x,-nx,v[-1]) for x,v in [[tt,t], [yy,y]]]
    if y[0] > y[argmax[0]]: tmax,ymax = blah(tmax,ymax)
    if y[0] < y[argmin[0]]: tmin,ymin = blah(tmin,ymin)
    if y[-1] > y[argmax[-1]]: tmax,ymax = blah2(tmax,ymax)
    if y[-1] < y[argmin[-1]]: tmin,ymin = blah2(tmin,ymin)
    
    #compute spline enevlopes and mean
    spline_min, spline_max = map(interp1d, [tmin,tmax], [ymin,ymax], ['cubic']*2)
    m = (spline_min(t) + spline_max(t))/2.0
    h = y - m

    plt.plot(t,y,'-',t,m,'-')
    plt.plot(tmin,ymin,'k.',tmax,ymax,'k.')
    tmin = np.linspace(tmin[0],tmin[-1],1000)
    tmax = np.linspace(tmax[0],tmax[-1],1000)
    plt.plot(tmin,spline_min(tmin),'-r',tmax,spline_max(tmax),'r-')

    return h

def sift3(t,y):
    #identify the relative extrema
    argmin, argmax = argextrema(y)
    
    #if there are too few extrema, raise an exception
    n = 2
    if len(argmin) < n or len(argmax) < n:
        raise FlatFunction('Fewer than {} max or min in the series -- cannot sift.'.format(n))
    
    #function to mirror nearest two extrema about the end
    def reflect(i):
        #parse out the end and extrema points
        tend,tmin,tmax = [t[ii] for ii in [i,argmin[i],argmax[i]]]
        
        if abs(tmin-tend) < abs(tmax-tend): #if min is closest to the end            
            ymin = y[argmin[i]]
            tmax, ymax = t[i], y[i]
        else:
            ymax = y[argmax[i]]
            tmin, ymin = t[i], y[i]
            
        tmirrored = [tend + (tend - tmin), tend + (tend - tmax)]
        ymirrored = [ymin,ymax]
        return tmirrored, ymirrored
    
    #get the mirrored times and construct the vectors with extended ends
    [tleft, yleft], [tright, yright] = map(reflect, [0,-1])
    tmin, tmax, ymin, ymax = map(np.concatenate, ([[tleft[0]], t[argmin], [tright[0]]],
                                                  [[tleft[1]], t[argmax], [tright[1]]],
                                                  [[yleft[0]], y[argmin], [yright[0]]],
                                                  [[yleft[1]], y[argmax], [yright[1]]]))
    
    #compute spline enevlopes and mean
    spline_min, spline_max = map(interp1d, [tmin,tmax], [ymin,ymax], ['cubic']*2)
    m = (spline_min(t) + spline_max(t))/2.0
    h = y - m
    
    return h

def sift2(t,y):
    #identify the relative extrema
    argmin, argmax = argextrema(y)
    
    #if there are too few extrema, raise an exception
    n = 2
    if len(argmin) < n or len(argmax) < n:
        raise FlatFunction('Fewer than {} max or min in the series -- cannot sift.'.format(n))
    
    #function to mirror nearest two extrema about the end
    nref = 4
    def reflect(i):
        #parse out the end and extrema points
        [tend,yend],[tmin,ymin],[tmax,ymax] = [[t[ii],y[ii]] for ii in 
                                               [i,argmin[i],argmax[i]]]
        
        #mirror the points about the end if it is outside of the two extrema
        #else mirror about the extremum nearest the end
        if yend > ymax or yend < ymin: #if the end is outside the relative extrema
            taxis = tend
            if yend > ymax: tmax, ymax = tend,yend
            if yend < ymin: tmin, ymin = tend,yend
        else:
            i2 = i+1 if i >= 0 else i-1 #second index from end (+1 if left, -1 right)
            if abs(tmin-tend) < abs(tmax-tend): #if min is closest to the end
                taxis = tmin
                ii = argmin[i2]
                tmin, ymin = t[ii], y[ii]
            else:
                taxis = tmax
                ii = argmax[i2]
                tmax, ymax = t[ii], y[ii]
            
        tmirrored = [taxis + (taxis - tmin), taxis + (taxis - tmax)]
        ymirrored = [ymin,ymax]
        return tmirrored, ymirrored
    
    #get the mirrored times and construct the vectors with extended ends
    [tleft, yleft], [tright, yright] = map(reflect, [0,-1])
    tmin, tmax, ymin, ymax = map(np.concatenate, ([tleft[0], t[argmin], tright[0]],
                                                  [tleft[1], t[argmax], tright[1]],
                                                  [yleft[0], y[argmin], yright[0]],
                                                  [yleft[1], y[argmax], yright[1]]))
    
    #compute spline enevlopes and mean
    spline_min, spline_max = map(interp1d, [tmin,tmax], [ymin,ymax], ['cubic']*2)
    m = (spline_min(t) + spline_max(t))/2.0
    h = y - m
    
#    plt.plot(t,y,'-',t,m,'-')
#    plt.plot(tmin,ymin,'k.',tmax,ymax,'k.')
#    tmin = np.linspace(tmin[0],tmin[-1],1000)
#    tmax = np.linspace(tmax[0],tmax[-1],1000)
#    plt.plot(tmin,spline_min(tmin),'-r',tmax,spline_max(tmax),'r-')
    
    return h
        
def sift(t,y):
    #identify the relative extrema
    argmin, argmax = argextrema(y)
    
    #if there are less than two extrema, raise an exception
    if len(argmin) + len(argmax) < 2:
        raise ValueError('Fewer than two extrema in the series -- cannot sift.')
    
    #create splines
    def spline(i):
        #reflect the first/last extrema at the beginning/end of the series
        #note that attemping a fixed-slope end condition gave divergent results
        #on simulated noisy data
        tbeg,tend = t[0] + (t[0] - t[i[0]]), t[-1] + (t[-1] - t[i[-1]])
        text = np.concatenate([[tbeg],t[i],[tend]])
        yext = np.concatenate([[y[i[0]]],y[i],[y[i[-1]]]])
        #create spline function
        spline = interp1d(text,yext,'cubic')
        return spline
    
    spline_min, spline_max = map(spline, [argmin,argmax])
    
    #compute mean
    m = (spline_min(t) + spline_max(t))/2.0
    h = y - m
    return h