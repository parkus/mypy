# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 11:43:52 2014

@author: Parke
"""
import numpy as np
import pdb

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
    if type(ary) != np.ndarray: ary = np.array(ary)
    if ary.ndim == 1: ary.resize([1,len(ary)])    
    ivec = np.digitize(ary[keyrow,:], bins)
    divvied = [ary[:, ivec == i] for i in np.arange(1,len(bins))]
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
    if unsorted: vec.sort()
    wtd = weights != None
    
    edges, count = [(vec[0] + vec[1])/2.0], []
    i, iold = chunksize, 0
    while i+1 < len(vec)-1:
        #if the endpoint and following point have the same value
        while vec[i] == vec[i+1]:
            i += 1
            if i+1 >= len(vec)-1:
                edges.append(vec[-1])
                count.append(np.sum(weights[iold:i]) if wtd else (i - iold))
                break
        edges.append((vec[i] + vec[i+1])/2.0)
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
    if unsorted: vec.sort()
        
    #use indices bc if len(vec) % chunksize == 0 there will be a point left of
    #the last bin edges, but not right of it
    iright = np.arange(1, len(vec), chunksize)
    
    edges = (vec[iright] + vec[iright-1])/2.0
    return edges
    
def chunk_sum(vec, chunksize):
    """Computes the sum of even numbered chunks of points in a vector.
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
    delta = y[1:] - y[:-1]
    pos_neg = delta//abs(delta)
    curve_sign = pos_neg[1:] - pos_neg[:-1]
    argmax = np.nonzero(curve_sign < 0)[0] + 1
    argmin = np.nonzero(curve_sign > 0)[0] + 1
    return argmin,argmax