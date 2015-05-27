# -*- coding: utf-8 -*-
"""
Created on Fri May 30 17:15:27 2014

@author: Parke
"""

import numpy as np
import matplotlib as mplot
import matplotlib.pyplot as plt
from mayavi import mlab
from color.maps import true_temp

dpi = 100
fullwidth = 10.0
halfwidth = 5.0

def tight_axis_limits(ax, xory='both', margin=0.05, datalim=None):

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
        if datalim == None: datalim = ax.dataLim.extents[[0,2]]
        axlim = ax.get_xlim()
        scale = ax.get_xscale()
        ax.set_xlim(newlim_either(datalim,axlim,scale))
    if xory == 'y' or xory == 'both':
        if datalim == None: datalim = ax.dataLim.extents[[1,3]]
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
    if not np.allclose(dx, dx[0], 1e-2) or not np.allclose(dy, dy[0], 1e-2):
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

def errorpoly(x, y, yerr, fmt=None, ecolor=None, ealpha=0.5, **kw):
    p = plt.plot(x,y,fmt, **kw)
    if len(yerr.shape) == 2:
        ylo = y - yerr[0,:]
        yhi = y + yerr[1,:]
    else:
        ylo, yhi = y - yerr, y + yerr
    if ecolor == None: ecolor = p[0].get_color()
    f = plt.fill_between(x,ylo,yhi,color=ecolor,alpha=ealpha)
    return p[0],f

def onscreen_pres(mpl, screenwidth=1200):
    """
    Set matplotlibrc values so that plots are readable as they are created
    and maximized for an audience far from a screen.

    Parameters
    ----------
    mpl : module
        Current matplotlib module. Use 'import matplotlib as mpl'.
    screewidth : int
        Width of the screen in question in pixels.

    Returns
    -------
    None
    """
    mpl.rcParams['lines.linewidth'] = 2
    fontsize = round(14 / (800.0 / screenwidth))
    mpl.rcParams['font.size'] = fontsize

def stars3d(ra, dec, dist, T=5000.0, r=1.0, labels='', view=None):
    """
    Make a 3D diagram of stars positions relative to the Sun, with
    semi-accurate colors and distances as desired. Coordinates must be in
    degrees.

    Meant to be used with only a handful of stars.
    """
    n = len(ra)
    makearr = lambda v: np.array([v] * n) if np.isscalar(v) else v
    T, r, labels = map(makearr, (T, r, labels))

    # add the sun
    ra, dec, dist = map(np.append, (ra, dec, dist), (0.0, 0.0, 0.0))
    r, T, labels = map(np.append, (r, T, labels), (1.0, 5780.0, 'Sun'))

    # get xyz coordinates
    z = dist * np.sin(dec * np.pi / 180.0)
    h = dist * np.cos(dec * np.pi / 180.0)
    x = h * np.cos(ra * np.pi / 180.0)
    y = h * np.sin(ra * np.pi / 180.0)

    # make figure
    fig = mlab.figure(bgcolor=(0,0,0), fgcolor=(1,1,1), size=(800,800))

    # plot lines down to the dec=0 plane for all but the sun
    lines = []
    for x1, y1, z1 in zip(x, y, z)[:-1]:
        xx, yy, zz = [x1, x1], [y1, y1], [0.0, z1]
        line = mlab.plot3d(xx, yy, zz, color=(0.7,0.7,0.7), line_width=0.5,
                           figure=fig)
        lines.append(line)

    # plot spheres
    r_factor = np.max(dist) / 30.0
    pts = mlab.quiver3d(x, y, z, r, r, r, scalars=T, mode='sphere',
                        scale_factor=r_factor, figure=fig, resolution=100)
    pts.glyph.color_mode = 'color_by_scalar'

    # center the glyphs on the data point
    pts.glyph.glyph_source.glyph_source.center = [0, 0, 0]

    # set a temperature colormap
    cmap = true_temp(T)
    pts.module_manager.scalar_lut_manager.lut.table = cmap

    # set the camera view
    mlab.view(focalpoint=(0.0, 0.0, 0.0))
    if view is not None:
        mlab.view(*view)

    ## add labels
    # unit vec to camera
    view = mlab.view()
    az, el = view[:2]
    hc = np.sin(el * np.pi / 180.0)
    xc = hc * np.cos(az * np.pi / 180.0)
    yc = hc * np.sin(az * np.pi / 180.0)
    zc = -np.cos(el * np.pi / 180.0)

    # unit vec orthoganal to camera
    if xc**2 + yc**2 == 0.0:
        xoff = 1.0
        yoff = 0.0
        zoff = 0.0
    else:
        xoff = yc / np.sqrt(xc**2 + yc**2)
        yoff = np.sqrt(1.0 - xoff**2)
        zoff = 0.0

#    xoff, yoff, zoff = xc, yc, zc

    # scale orthogonal vec by sphere size
    r_label = 1.0 * r_factor
    xoff, yoff, zoff = [r_label * v for v in [xoff, yoff, zoff]]

    # plot labels
    size = r_factor / 2.0
    for xx, yy, zz, label in zip(x, y, z, labels):
        mlab.text3d(xx + xoff, yy + yoff, zz + zoff, label, figure=fig,
                    color=(0.5,0.5,0.5), scale=size)

    mlab.draw()
    return fig


