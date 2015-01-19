import numpy as np
cimport numpy as np

LONG = np.int64
ctypedef np.int64_t LONG_t
ctypedef np.float64_t DBL_t

def rebin_or(np.ndarray[DBL_t] nb, np.ndarray[DBL_t] ob, np.ndarray[LONG_t] ov):
    if nb.ndim > 1 or ob.ndim > 1 or ov.ndim > 1:
        raise ValueError('1D arrays only')
    
    cdef np.ndarray[LONG_t] binmap = np.searchsorted(ob, nb, 'left')
    
    cdef long n = nb.shape[0] - 1
    cdef np.ndarray[LONG_t] nv = np.zeros(n, dtype=LONG)
    cdef long k, i0, i1
    
    for k in range(n):
        i0 = binmap[k]
        i1 = binmap[k+1]
        if nb[k] != ob[i0] and i0 > 0:
            i0 = i0 - 1
        
        for i in range(i0, i1):
            nv[k] = nv[k] | ov[i]
    
    return nv