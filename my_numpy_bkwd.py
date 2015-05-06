# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:25:46 2015

@author: rolo7566
"""

def argextrema(y, separate=True):
    """
    Deprecated in favor of argrel{min|max} in scypy.signal to get separate
    extrema in about the same CPU time.

    If you need a list of
    all relative extrema in order, using this with separate=False takes about
    half the time as by combining the scipy
    functions with searchsorted.

    Returns the indices of the local extrema of a series. When consecutive
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