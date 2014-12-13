# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:46:02 2014

@author: Parke
"""
from numpy import median, sum, nan, array
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, norm
from math import sqrt
from my_numpy import block_edges

def runstest(x, divider=None, passfail=False):
    """
    Perform the Wald-Wolfowitz runs test.
    
    The Wald-Wolfowitz runs test evaluates the null-hypothesis that the data 
    are independent and identically distributed assuming a parametric 
    form for the underlying probability distribution function of the data [1].

    Parameters
    ----------
    x : 1-D array-like
        The data to be tested.
        
    divider : float, optional
        The value to use to assign the data binary values. Data above
        the divider value will be assigned a 1, data below the divider will be
        assigned a 0, and data equal to the divider will be omitted. 
        
        To test a function fit, supply runstest with the fit-subtracted data 
        and set divider to 0.0.
        
        If not supplied, the median value of the data is used.  
        
    passfail : {True|False}, optional
        If set to true, the function simply returns True if the test rejects
        the null hypothesis to > 95% confidence and False otherwise. This was
        implemented for short data series, where the runs statistic deviates
        significantly from a standard normal distribution and a lookup
        table is used to determine whether the test rejects the null hypothesis
        or not instead of computing a p-value. 
        
    Returns
    -------
    z : float
        The test statistic.
    p-value : float
        The two-tailed p-value (probability to exceed) for the hypothesis test.
    
    References
    ---------
    .. [1] http://www.itl.nist.gov/div898/handbook/eda/section3/eda35d.htm
    
    """
    #TODO: maybe add this to scipy one day...
    
    if divider is None:
        divider = median(x)
    x_culled = x[x != divider]
    
    # Identify the runs
    x_binary = (x_culled > divider)
    run_edges = ((x_binary[1:] - x_binary[:-1]) != 0)
    r = sum(run_edges) + 1
    n = len(x_binary)
    n1 = sum(x_binary)
    n2 = n - n1
    
    # Use an approximating normal distribution if there is sufficient data
    if n1 > 10 and n2 > 10:
        mu = 2*n1*n2/n + 1
        var = (mu-1)*(mu-2)/(n-1)
        z = abs(r - mu)/sqrt(var)
        if passfail:
            return z > 1.96
        else:
            p = 2*norm.cdf(z)
            return z, p
            
    # Otherwise, use table lookup
    else:
        if not passfail:
            raise ValueError("The data in x have too few runs to compute a "
                             "reliable and exact p-value. Consider using "
                             "the function in passfail mode, which uses a "
                             "table lookup to provide a result for any data.")
        else:
            if n1 < 2 and n2 < 2:
                return False
            
            # From Wall and Jenkins, Practical Statistics for Astronomers,
            # Cambridge University Press, 2003, pp 257, Table A2.8
            # FIXME: the lo array is symmetric and I believe the hi array
            #    should be as well. However, it is asymmetric in the reference.
            #    Need to check against another reference.
            hi = [[nan]*19,
                  [nan]*19,
                  [nan]*3 + [9,9] + [nan]*14,
                  [nan]*2 + [9,10,10,11,11] + [nan]*12,
                  [nan]*2 + [9,10,11,12,12,13,13,13,13] + [nan]*8,
                  [nan]*3 + [11,12,13,13,14,14,14,14,15,15,15] + [nan]*5,
                  [nan]*3 + [11,12,13,14,14,15,15,16,16,16,16,17,17,17,17,17],
                  [nan]*4 + [13,14,14,15,16,16,16,17,17,18,18,18,18,18,18],
                  [nan]*4 + [13,14,15,16,16,17,17,18,18,18,19,19,19,20,20],
                  [nan]*4 + [13,14,15,16,17,17,18,19,19,19,20,20,20,21,21],
                  [nan]*4 + [13,14,16,16,17,18,19,19,20,20,21,21,21,22,22],
                  [nan]*5 + [15,16,17,18,19,19,20,20,21,21,22,22,23,23],
                  [nan]*5 + [15,16,17,18,19,19,20,20,21,21,22,22,23,23],
                  [nan]*5 + [15,16,18,18,19,20,21,22,22,23,23,24,24,25],
                  [nan]*6 + [17,18,19,20,21,21,22,23,23,24,25,25,25],
                  [nan]*6 + [17,18,19,20,21,22,23,23,24,25,25,26,26],
                  [nan]*6 + [17,18,19,20,21,22,23,24,25,25,26,26,27],
                  [nan]*6 + [17,18,20,21,22,23,23,24,25,25,26,26,27],
                  [nan]*6 + [17,18,20,21,22,23,24,25,25,26,27,27,28]]
            lo = [[nan]*10 + [2]*9,
                  [nan]*4 + [2]*9 + [3]*6,
                  [nan]*3 + [2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4],
                  [nan,nan,2,2,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5],
                  [nan,2,2,3,3,3,3,4,4,4,4,5,5,5,5,5,5,6,6],
                  [nan,2,2,3,3,3,4,4,5,5,5,5,5,6,6,6,6,6,6],
                  [nan,2,3,3,3,4,4,5,5,5,6,6,6,6,6,7,7,7,7],
                  [nan,2,3,3,4,4,5,5,5,6,6,6,7,7,7,7,8,8,8],
                  [nan,2,3,3,4,5,5,5,6,6,7,7,7,7,8,8,8,8,9],
                  [nan,2,3,4,4,5,5,6,6,7,7,7,8,8,8,9,9,9,9],
                  [2,2,3,4,4,5,6,6,7,7,7,8,8,8,9,9,9,10,10],
                  [2,2,3,4,5,5,6,6,7,7,8,8,9,9,9,10,10,10,10],
                  [2,2,3,4,5,5,6,7,7,8,8,9,9,9,10,10,10,11,11],
                  [2,3,3,4,5,6,6,7,7,8,8,9,9,10,10,11,11,11,12],
                  [2,3,4,4,5,6,6,7,8,8,9,9,10,10,11,11,11,12,12],
                  [2,3,4,4,5,6,7,7,8,9,9,10,10,11,11,11,12,12,13],
                  [2,3,4,5,5,6,7,8,8,9,9,10,10,11,11,12,12,13,13],
                  [2,3,4,5,6,6,7,8,8,9,10,10,11,11,12,12,13,13,13],
                  [2,3,4,5,6,6,7,8,9,9,10,10,11,12,12,13,13,13,14]]
            hi, lo = map(array, [hi, lo])
            return (r > hi[n1,n2] or r < lo[n1,n2])
            
def flag_anomalies(x, test='runs', metric='chi2', tol=0.05, trendfit='median', 
                   plotsteps=False, maxiter=1000):
    """
    Identifies groups of statistically anomalous data.
    
    The data are evaluated using the provided test to determine the sigma-level
    at which the test hypothesis may be rejected (e.g., statistical 
    independence via the runs test, consistency with normal function via
    shapiro-wilks test). If the data fail the test, data are grouped into
    "runs" (succesive points above or below the sample mean). The deviance of
    each run from the test is quantified via the method or function specified
    by metric (e.g. chi-square value assuming the data have error equal to the
    sample standard deviation). The run with the greatest deviance is flagged
    and the statistical test is rerun on the unflagged data. This process is
    iterated until the unflagged data are consistent with the test hypothesis
    to within the sigma-level specified by tol.
    
    To deal with the changing extent of the runs as the sample mean changes,
    after each iteration points at the edges of each run are added or removed
    if their position relative to the sample mean has changed.
    
    Parameters
    ----------
    data : list or 1-D numpy array
        The data series to be analyzed for anomalies.
    
    test : string or function
        Tests the data against a null-hypothesis given the trend-removed data
        and a boolean vector identifying unflagged data points.  
    
        The statistical test to use for identifying anomalies. This can be a 
        string specifying a pre-defined test or a function object. The ability 
        to supply a user-defined function enables such things as fitting and 
        removing a trend from the data prior to applying a statistical test. 
        Functions must accept only a 1-D boolean numpy array identifying the
        flagged data (i.e. the function must know about the data independent of
        flag_anomalies). This is because the user may want to include data,
        errors, independent data, etc. in the function, in which case the
        unflagged values of each can be consistently selected using flags as a 
        boolean index. The test function must return the probability to exceed 
        for the null hypothesis it tests.
        
        Built in tests currently include:
            - 'shapiro-wilks'|'sw' : Tests whether the data are normally
                distributed.
            - 'runs'|'r' : Tests whether the data are independent.
          
    metric : string or function
        Computes a deviation value for each data point after trend removal. 
        These points are summed by run to compute the anomalousness of each
        run and decide which is the next run that should be removed.
        
        Built in metrics currently include:
            - 'len' : the number of points in the run
            - 'chi2' : sum(data[run] - mean(data[unflagged])) -- like
                chi-square with constant variance
            - 'area' : absolute value of the integral of the mean-subtracted 
                values
        
    tol : float
        The probability to exceed at which to accept the unflagged data as containing no
        anomalies. When the test returns a value below this level, iteration
        stops.
    
    trendfit: string or functions
        Fits a trend to the data and returns the value of the trend at each pt.
        
    plotsteps: boolean or function
        If True, the data are plotted after each iteration along with the
        value returned from test. This allows the user to decide on an 
        appropriate value for tol. As with test, a function can be supplied
        that accepts flags as an input to customize the plotting. Otherwise
        the data are simply plotted versus their index. 
                
    Returns
    -------
    flags : 1-D numpy boolean array 
        Vector of True or False flags identifying anomalous points with True.
        
    Modification History
    --------------------
    2014-10-23 written by Parke Loyd
    """
    
    # PARSE THE INPUT
    #-------------------------------------------------------------------------- 
    builtintests = {'runs':runstest, 'r':runstest, 'shapiro-wilks':shapiro,
                    'shapiro':shapiro, 'sw':shapiro}
    if type(test) is str: 
        testfunc = builtintests[test.lower()]
        def test(x,flags): 
            good = np.logical_not(flags)
            return testfunc(x[good])[1]
    
    builtintrends = {'median' : np.median, 'mean':np.mean}
    if type(trendfit) is str: 
        fitfunc = builtintrends[trendfit.lower()]
        def trendfit(x,good): 
            value = fitfunc(x[good])
            return np.ones(x.shape)*value
    
    builtinmetrics = {'chi2' : (lambda x: x**2), 
                      'area' : (lambda x: abs(x)),
                      'len'  : (lambda x: np.ones(x.shape))}
    if type(metric) is str: metric = builtinmetrics[metric.lower()]
    
    if plotsteps is True:
        def plotsteps(x,good):
            ind = np.arange(len(x))
            plt.plot(ind, x, 'r.')
            plt.plot(ind[good],x[good], 'g.')
    
    n = len(x)
    
    #ITERATIVELY IDENTIFY ANOMALIES
    #--------------------------------------------------------------------------    
    good = np.ones(n, bool)
    counter = 0
    while True:
        #fit the retained data
        fit = trendfit(x, good)
        x1 = x - fit
        
        #adjust previous groups based on new trend through good data
        #----------------------------------------------------------------------
        pospts = (x1 > 0)
        flags = np.logical_not(good)
        if np.sum(flags) > 0:
            begs,ends = block_edges(flags)
            for i0,i1 in zip(begs,ends):
                #is the peak of the anomaly positive?
                relargmax = np.argmax(abs(x1[i0:i1]))
                absargmax = relargmax + i0
                pos = pospts[absargmax]
                
                #delete the anomaly
                good[i0:i1] = True
                
                #"creep" the edges of the anomaly in or out according to its new
                #position relative to the trend
                #TODO: cythonize?
                while i0-1 > 0 and pospts[i0-1] == pos: i0 -= 1
                while i0+1 < i1 < n and pospts[i0+1] != pos: i0 += 1
                while i1-2 > i0 > 0 and pospts[i1-2] != pos: i1 -= 1
                while i1 < n and pospts[i1] == pos: i1 += 1
                    
                #reset the anomaly with adjusted edges
                good[i0:i1] = False
        
        #check if statistical test is passed
        p = test(x,flags)
        if p > tol:
            return flags
        
        #plot if desired
        if plotsteps is not False:
            plt.ioff()
            plotsteps(x)
            plt.show()
            stop = raw_input("p = {}\nStop? ('y' to stop, anything to "
                             "continue)".format(p))
            if stop == 'n': return
                
        #identify next largest anomaly
        #----------------------------------------------------------------------
        ptdevs = metric(x1)
        edges = [0] + list(np.nonzero(pospts[1:] - pospts[:-1])[0] + 1) + [n]
        runs = zip(edges[:-1],edges[1:])
        deviations = np.zeros(len(runs))
        for i,run in enumerate(runs):
            j0,j1 = run
            mid = (j0 + j1)/2
            deviations[i] = np.sum(ptdevs[j0:j1]) if good[mid] else 0.0
        anmly = np.argmax(deviations)
        j0,j1 = runs[anmly]
        good[j0:j1] = False
        
        counter += 1
        if counter > maxiter:
            raise ValueError('Exceeded allowable number of iterations.')