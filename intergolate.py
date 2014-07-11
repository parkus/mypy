# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 11:43:52 2014

@author: Parke
"""
__all__ = ['intergolate']

def intergolate(x_bin_edges,xin,yin):
    """***Merged into my_numpy on 2014/05/21*** Compute average of xin,yin within supplied bins.
    
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
    
    #define variables to stor the indices of the points just right of the left
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

        
        