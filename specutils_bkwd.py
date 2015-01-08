# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 11:58:26 2014

@author: Parke
"""
import numpy as np
import matplotlib.pyplot as plt

def identify_continuum(wbins, y, err, function_generator, maxsig=2.0, 
                       emission_weight=1.0, maxcull=0.99, plotsteps=False):
    """
    Retained for backwards compatability. Use splitspec now.
    """
    if type(plotsteps) == bool:
        plotbelow = np.inf if plotsteps else 0.0
    else:
        plotbelow = plotsteps
        plotsteps = True
    if plotsteps: plt.ioff()
    if len(wbins) != len(y):
        raise ValueError('The shape of wbins must be [len(y), 2]. These '
                         'represent the edges of the wavelength bins over which '
                         'photons were counted (or flux was integrated).')
    wbins, y, err = map(np.array, [wbins, y, err])
    Npts = len(y)
    
    if plotsteps:
        wave = (wbins[:,0] + wbins[:,1])/2.0
        wbinsin, wavein, yin= wbins, wave, y
        
    while True:
        #fit to the retained data
        f = function_generator(wbins, y, err)
        
        #count the runs
        expected = f(wbins)
        posneg = (y > expected)
        run_edges = ((posneg[1:] - posneg[:-1]) !=0)
        Nruns = np.sum(run_edges) + 1
        
        #compute the PTE for the runs test
        N = len(y)
        Npos = np.sum(posneg)
        Nneg = N - Npos
        mu = 2*Npos*Nneg/N + 1
        var = (mu-1)*(mu-2)/(N-1)
        sigruns = abs(Nruns - mu)/np.sqrt(var)
        
        #if the fit passes the runs test, then return the good wavelengths
        if sigruns < maxsig:
            non_repeats = (wbins[:-1,1] != wbins[1:,0])
            w0, w1 = wbins[1:,0][non_repeats], wbins[:-1,1][non_repeats]
            w0, w1 = np.insert(w0, 0, wbins[0,0]), np.append(w1, wbins[-1,1])
            return np.array([w0,w1]).T
        else:
            #compute the chi2 PTE for each run
            iedges = np.concatenate([[0], np.nonzero(run_edges)[0]+1, [len(run_edges)+1]]) #the slice index
            chiterms = ((y - expected)/err)**2
            chisum = np.cumsum(chiterms)
            chiedges = np.insert(chisum[iedges[1:]-1], 0, 0.0)
            chis =  chiedges[1:] - chiedges[:-1]
            DOFs = (iedges[1:] - iedges[:-1])
            sigs = abs(chis - DOFs)/np.sqrt(2*DOFs)
            if emission_weight != 1.0:
                if posneg[0] > 0: sigs[::2] = sigs[::2]*emission_weight
                else: sigs[1::2] = sigs[1::2]*emission_weight
            
            good = np.ones(len(y), bool)            
            good[np.argmax(sigs)] = False #mask out the run with the smallest PTE
            keep = np.concatenate([[g]*d for g,d in zip(good, DOFs)]) #make boolean vector
            
            if plotsteps and (sigruns < plotbelow):
                plt.plot(wavein, yin)
                plt.plot(wavein,f(wbinsin),'k')
                plt.plot(wave,y,'g.')
                trash = np.logical_not(keep)
                plt.plot(wave[trash], y[trash], 'r.') 
                ax = plt.gca()
                plt.text(0.8, 0.9, '{}'.format(sigruns), transform=ax.transAxes)
                plt.show()
            if plotsteps: wave = wave[keep]
            wbins, y, err = wbins[keep], y[keep], err[keep] 
        
        if float(len(y))/Npts < (1.0 - maxcull):
            raise ValueError('More than maxcull={}% of the data has been '
                             'removed, but the remaining data and associated '
                             'fit is still not passing the Runs Test. Consider '
                             'checking that the fits are good, relaxing '
                             '(increasing) the maxsig condition for passing '
                             'the Runs Test, or increasing maxcull to allow '
                             'more data to be masked out.')