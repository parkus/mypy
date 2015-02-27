import numpy as np
cimport numpy as np

LONG = np.int64
DBL = np.float64
INT = np.int32
ctypedef np.int64_t LONG_t
ctypedef np.float64_t DBL_t
ctypedef np.int32_t INT_t

def rebin(np.ndarray[DBL_t] nb, np.ndarray[DBL_t] ob, ov, method):
    n = len(nb) - 1
    if type(ov.item(0)) is float:
        return rebin_float(nb, ob, ov.astype('f8'), method, n)
    elif type(ov.item(0)) is int:
        return rebin_int(nb, ob, ov.astype('i4'), method, n)
    elif type(ov.item(0)) is long:
        return rebin_long(nb, ob, ov.astype('i8'), method, n)

def rebin_float(nb, ob, np.ndarray[DBL_t] ov, method, n):
    cdef np.ndarray[DBL_t] nv = np.zeros(n, dtype=DBL)
    return __rebin(nb, ob, ov, nv, method, n)

def rebin_long(nb, ob, np.ndarray[LONG_t] ov, method, n):
    cdef np.ndarray[LONG_t] nv = np.zeros(n, dtype=LONG)
    return __rebin(nb, ob, ov, nv, method, n)

def rebin_int(nb, ob, np.ndarray[INT_t] ov, method, n):
    cdef np.ndarray[INT_t] nv = np.zeros(n, dtype=INT)
    return __rebin(nb, ob, ov, nv, method, n)

def __rebin(nb, ob, ov, nv, method, n):

    cdef np.ndarray[LONG_t] binmap = np.searchsorted(ob, nb, 'left')

    cdef long k, i0, i1

    #guess for speed I won't make it check what method to use in each loop iteration
    if method == 'sum':
        for k in range(n):
            i0 = binmap[k]
            i1 = binmap[k+1]
            if i0 == i1:
                nv[k] = (nb[k+1] - nb[k]) / (ob[i0] - ob[i0-1]) * ov[i0-1]
            else:
                left, right, mid = 0.0, 0.0, 0.0
                if nb[k] != ob[i0]:
                    left = (ob[i0] - nb[k]) / (ob[i0] - ob[i0-1]) * ov[i0-1]
                if nb[k+1] != ob[i1]:
                    right = (nb[k+1] - ob[i1-1]) / (ob[i1] - ob[i1-1]) * ov[i1-1]
                    i1 -= 1
                mid = 0.0
                for i in range(i0, i1): mid += ov[i]
                nv[k] = left + mid + right
    elif method == 'or':
        for k in range(n):
            i0 = binmap[k]
            i1 = binmap[k+1]
            if nb[k] != ob[i0] and i0 > 0:
                i0 = i0 - 1

            for i in range(i0, i1):
                nv[k] = nv[k] | ov[i]
    elif method == 'min':
        for k in range(n):
            i0 = binmap[k]
            i1 = binmap[k+1]
            if nb[k] != ob[i0] and i0 > 0:
                i0 = i0 - 1

            nv[k] = ov[i0]
            for i in range(i0+1, i1):
                if ov[i] < nv[k]:
                    nv[k] = ov[i]
    elif method == 'max':
        for k in range(n):
            i0 = binmap[k]
            i1 = binmap[k+1]
            if nb[k] != ob[i0] and i0 > 0:
                i0 = i0 - 1

            nv[k] = ov[i0]
            for i in range(i0+1, i1):
                if ov[i] > nv[k]:
                    nv[k] = ov[i]

    return nv