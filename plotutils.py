# -*- coding: utf-8 -*-
"""
Created on Fri May 30 17:15:27 2014

@author: Parke
"""

import numpy as np
import matplotlib as mplot
import matplotlib.pyplot as plt

dpi = 100
fullwidth = 10.0
halfwidth = 5.0

def tight_axis_limits(ax, xory='both', margin=0.05):
    
    def newlim(oldlim):
        delta = abs(oldlim[1] - oldlim[0])
        pad = delta*margin
        if oldlim[1] > oldlim[0]:
            return (oldlim[0] - pad, oldlim[1] + pad)
        else:
            return (oldlim[0] + pad, oldlim[1] - pad)
    
    def newlim_log(oldlim):
        loglim = [np.log10(l) for l in oldlim]
        newloglim = newlim(loglim)
        return (10.0**newloglim[0], 10.0**newloglim[1])
    
    def newlim_either(oldlim,axlim,scale):
        if axlim[1] < axlim [0]: oldlim = oldlim[::-1]
        if scale == 'linear':
            return newlim(oldlim)
        elif scale == 'log':
            return newlim_log(oldlim)
        elif scale == 'symlog':
            print 'Past Parke to future Parke, you did\'t write an implementation for symlog scaled axes.'
            return oldlim
    
    if xory == 'x' or xory == 'both':
        datalim = ax.dataLim.extents[[0,2]]
        axlim = ax.get_xlim()
        scale = ax.get_xscale()
        ax.set_xlim(newlim_either(datalim,axlim,scale))
    if xory == 'y' or xory == 'both':
        datalim = ax.dataLim.extents[[1,3]]
        axlim = ax.get_ylim()
        scale = ax.get_yscale()
        ax.set_ylim(newlim_either(datalim,axlim,scale))
        
#TODO: discard this function?
def standard_figure(app, slideAR=1.6, height=1.0):
    """Generate a figure of standard size for publishing.
    
    implemented values for app (application) are:
    'fullslide'
    
    height is the fractional height of the figure relative to the "standard"
    height. For slides the standard is the full height of a slide.
    
    returns the figure object and default font size
    """
    
    if app == 'fullslide':
        fontsize = 20
        figsize = [fullwidth, fullwidth/slideAR*height]
        fig = mplot.pyplot.figure(figsize=figsize, dpi=dpi)
    
    mplot.rcParams.update({'font.size': fontsize})
    return fig, fontsize        
    
def pcolor_reg(x, y, z, **kw):
    """
    Similar to `pcolor`, but assume that the grid is uniform,
    and do plotting with the (much faster) `imshow` function.

    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y should be 1-dimensional")
    if z.ndim != 2 or z.shape != (y.size, x.size):
        raise ValueError("z.shape should be (y.size, x.size)")
    dx = np.diff(x)
    dy = np.diff(y)
    if not np.allclose(dx, dx[0]) or not np.allclose(dy, dy[0]):
        raise ValueError("The grid must be uniform")

    if np.issubdtype(z.dtype, np.complexfloating):
        zp = np.zeros(z.shape, float)
        zp[...] = z[...]
        z = zp

    plt.imshow(z, origin='lower',
               extent=[x.min(), x.max(), y.min(), y.max()],
               interpolation='nearest',
               aspect='auto',
               **kw)
    plt.axis('tight')