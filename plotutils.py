# -*- coding: utf-8 -*-
"""
Created on Fri May 30 17:15:27 2014

@author: Parke
"""

import numpy as np
import matplotlib as mplot
import matplotlib.pyplot as plt
import mypy.my_numpy as mnp

dpi = 100
fullwidth = 10.0
halfwidth = 5.0

# use these with line.set_dashes and iterate through more linestyles than come with matplotlib
# consider ussing a ::2 slice for fewer
dashes = [[],
          [30, 10],
          [20, 8],
          [10, 5],
          [3, 2],
          [30, 5, 3, 5, 10, 5, 3, 5],
          [15] + [5, 3]*3 + [5],
          [15] + [5, 3]*2 + [5],
          [15] + [5, 3] + [5]]


def common_axes(fig, pos=None):
    if pos is None:
        bigax = fig.add_subplot(111)
    else:
        bigax = fig.add_axes(pos)
    [bigax.spines[s].set_visible(False) for s in ['top', 'bottom', 'left', 'right']]
    bigax.tick_params(labelleft=False, labelbottom=False, left='off', bottom='off')
    bigax.set_zorder(-10)
    return bigax


def log_frac(x, frac):
    l0, l1 = map(np.log10, x)
    ld = l1 - l0
    l = ld*frac + l0
    return 10**l


def log2linear(x, errneg=None, errpos=None):
    xl = 10**x
    result = [xl]
    if errneg is not None:
        xn = xl - 10**(x - np.abs(errneg))
        result.append(xn)
    if errpos is not None:
        xp = 10**(x + errpos) - xl
        result.append(xp)
    return result


def linear2log(x, errneg=None, errpos=None):
    xl = np.log10(x)
    result = [x]
    if errneg is not None:
        xn = xl - np.log10(x - np.abs(errneg))
        result.append(xn)
    if errpos is not None:
        xp = np.log10(x + errpos) - xl
        result.append(xp)
    return result


def step(*args, **kwargs):
    edges, values = args[0], args[1]

    # deal with potentially gappy 2-column bin specifications
    edges = np.asarray(edges)
    if edges.ndim == 2:
        if np.any(edges[1:,0] < edges[:-1,1]):
            raise ValueError('Some bins overlap')
        if np.any(edges[1:,0] < edges[:-1,0]):
            raise ValueError('Bins must be in increasing order.')
        gaps = edges[1:,0] > edges[:-1,1]
        edges = np.unique(edges)
        if np.any(gaps):
            values = np.insert(values, np.nonzero(gaps), np.nan)

    edges = mnp.lace(edges[:-1], edges[1:])
    values = mnp.lace(values, values)
    args = list(args)
    args[0], args[1] = edges, values
    ax = kwargs.pop('ax', plt.gca())
    return ax.plot(*args, **kwargs)


def point_along_line(x, y, xfrac=None, xlbl=None, scale='linear'):
    if scale == 'log':
        lx, ly = point_along_line(np.log10(x), np.log10(y), xfrac, xlbl, ylbl, scale)
        return 10 ** lx, 10 ** ly
    if xfrac is not None:
        if xfrac == 0:
            return x[0], y[0]
        if xfrac == 1:
            return x[-1], y[-1]
        else:
            d = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
            d = np.insert(d, 0, 0)
            f = d/d[-1]
            xp, yp = [np.interp(xfrac, f, a) for a in [x,y]]
            return xp, yp
    if xlbl is not None:
        return xlbl, np.interp(xlbl, x, y)


def textSize(ax_or_fig=None, coordinate='data'):
    """
    Return x & y scale factors for converting text sizes in points to another coordinate. Useful for properly spacing
    text labels and such when you need to know sizes before the text is made (otherwise you can use textBoxSize).

    Coordinate can be 'data', 'axes', or 'figure'.

    If data coordinates are requested and the data is plotted on a log scale, then the factor will be given in dex.

    """
    if ax_or_fig is None:
        fig = plt.gcf()
        ax = fig.gca()
    else:
        if isinstance(ax_or_fig, plt.Figure):
            fig = ax_or_fig
            ax = fig.gca()
        elif isinstance(ax_or_fig, plt.Axes):
            ax = ax_or_fig
            fig = ax.get_figure()
        else:
            raise TypeError('ax_or_fig must be a Figure or Axes instance, if given.')

    w_fig_in, h_fig_in = ax.get_figure().get_size_inches()

    if coordinate == 'fig':
        return 1.0/(w_fig_in*72), 1.0/(h_fig_in*72)

    w_ax_norm, h_ax_norm = ax.get_position().size
    w_ax_in = w_ax_norm * w_fig_in
    h_ax_in = h_ax_norm * h_fig_in
    w_ax_pts, h_ax_pts = w_ax_in*72, h_ax_in*72

    if coordinate == 'axes':
        return 1.0/w_ax_pts, 1.0/h_ax_pts

    if coordinate == 'data':
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if ax.get_xscale() == 'log': xlim = np.log10(xlim)
        if ax.get_yscale() == 'log': ylim = np.log10(ylim)
        w_ax_data = xlim[1] - xlim[0]
        h_ax_data = ylim[1] - ylim[0]
        return w_ax_data/w_ax_pts, h_ax_data/h_ax_pts


def tight_axis_limits(ax=None, xory='both', margin=0.05):
    if ax is None: ax = plt.gca()

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
            raise NotImplementedError('Past Parke to future Parke, you did\'t write an implementation for symlog'
                                      'scaled axes.')

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

def errorpoly(x, y, yerr, fmt=None, ecolor=None, ealpha=0.5, ax=None, **kw):
    if ax is None: ax = plt.gca()
    p = ax.plot(x, y, **kw) if fmt is None else ax.plot(x, y, fmt, **kw)
    if len(yerr.shape) == 2:
        ylo = y - yerr[0,:]
        yhi = y + yerr[1,:]
    else:
        ylo, yhi = y - yerr, y + yerr
    if ecolor is None: ecolor = p[0].get_color()

    # deal with matplotlib sometimes not showing polygon when it extends beyond plot range
    xlim = ax.get_xlim()
    inrange = mnp.inranges(x, xlim)
    if not np.all(inrange):
        n = np.sum(inrange)
        yends = np.interp(xlim, x, y)
        yloends = np.interp(xlim, x, ylo)
        yhiends = np.interp(xlim, x, yhi)
        x = np.insert(x[inrange], [0, n], xlim)
        y = np.insert(y[inrange], [0, n], yends)
        ylo = np.insert(ylo[inrange], [0, n], yloends)
        yhi = np.insert(yhi[inrange], [0, n], yhiends)

    f = ax.fill_between(x,ylo,yhi,color=ecolor,alpha=ealpha)
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


def textBoxSize(txt, transformation=None, figure=None):
    """Get the width and height of a text object's bounding box transformed to the desired coordinates. Defaults to
    figure coordinates if transformation is None."""

    fig= txt.get_figure() if figure is None else figure
    if transformation is None:
        transformation = fig.transFigure

    coordConvert = transformation.inverted().transform
    bboxDisp = txt.get_window_extent(fig.canvas.renderer)
    bboxConv = coordConvert(bboxDisp)
    w = bboxConv[1,0] - bboxConv[0,0]
    h = bboxConv[1,1] - bboxConv[0,1]
    return w, h


def stars3d(ra, dec, dist, T=5000.0, r=1.0, labels='', view=None, size=(800,800), txt_scale=1.0):
    """
    Make a 3D diagram of stars positions relative to the Sun, with
    semi-accurate colors and distances as desired. Coordinates must be in
    degrees. Distance is assumed to be in pc (for axes labels).

    Meant to be used with only a handful of stars.
    """
    from mayavi import mlab
    from color.maps import true_temp

    n = len(ra)
    dec, ra = dec*np.pi/180.0, ra*np.pi/180.0
    makearr = lambda v: np.array([v] * n) if np.isscalar(v) else v
    T, r, labels = map(makearr, (T, r, labels))

    # add the sun
    ra, dec, dist = map(np.append, (ra, dec, dist), (0.0, 0.0, 0.0))
    r, T, labels = map(np.append, (r, T, labels), (1.0, 5780.0, 'Sun'))

    # get xyz coordinates
    z = dist * np.sin(dec)
    h = dist * np.cos(dec)
    x = h * np.cos(ra)
    y = h * np.sin(ra)

    # make figure
    fig = mlab.figure(bgcolor=(0,0,0), fgcolor=(1,1,1), size=size)

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
    mlab.view(focalpoint=(0.0, 0.0, 0.0), figure=fig)
    if view is not None:
       mlab.view(*view, figure=fig)

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
    size = r_factor * txt_scale * 0.75
    for xx, yy, zz, label in zip(x, y, z, labels):
       mlab.text3d(xx + xoff, yy + yoff, zz + zoff, label, figure=fig,
                   color=(1,1,1), scale=size)

    ## add translucent dec=0 surface
    n = 101
    t = np.linspace(0.0, 2*np.pi, n)
    r = np.max(dist * np.cos(dec))
    x, y = r*np.cos(t), r*np.sin(t)
    z = np.zeros(n+1)
    x, y = [np.insert(a, 0, 0.0) for a in [x,y]]
    triangles = [(0, i, i + 1) for i in range(1, n)]
    mlab.triangular_mesh(x, y, z, triangles, color=(1,1,1), opacity=0.3, figure=fig)

    ## add ra=0 line
    line = mlab.plot3d([0, r], [0, 0], [0, 0], color=(1,1,1), line_width=1, figure=fig)
    rtxt = '{:.1f} pc'.format(r)
    orientation=np.array([180.0, 180.0, 0.0])
    mlab.text3d(r, 0, 0, rtxt, figure=fig, scale=size*1.25, orient_to_camera=False, orientation=orientation)

    if view is not None:
       mlab.view(*view, figure=fig)

    return fig


