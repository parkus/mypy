# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:46:02 2014

@author: Parke
"""
from numpy import median, sum, nan, array
import numpy as np
from scipy.stats import shapiro, norm, skewtest, poisson
from math import sqrt
from mypy.my_numpy import splitsum
from scipy.optimize import minimize as _minimize
from scipy.integrate import quad as _quad

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
            p = 2*(1 - norm.cdf(z))
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

def clean(x, tol, test='runs', metric='chi2', trendfit='median', maxiter=1000, printsteps=False):
    """
    Sequentially removes groups of anomalous data until the remaining data
    pases a user-specified test within some tolerance.

    Parameters
    ----------
    x : list or 1-D numpy array
        The data series to be analyzed for anomalies.

    tol : float
        Tolerance. When the value returned by testing the cleaned data is below
        this level, the data are considered successfully cleaned.

    test : {function|string}, optional
        Measures the "cleanliness" of the data, generally by computing the
        probability that the data could have been generated by some expected
        distribution (e.g. the normal distribution).

        String input can be used to specify one of the following built-in
        tests:
            - 'deviation size'|'devsize' : Value of the maximum deviation as
                quantified by "metric." Useful for simply removing all
                anomalies that exceed some threshold. Use tol to set that
                threshold.
            - 'shapiro-wilks'|'sw' : Tests whether the data are normally
                distributed. See scipy.stats.shapiro.
            - 'runs'|'r' : Tests whether the data are independent.
            - 'skew' : Tests whether data skewness is consistent with a normal
                distribution. See scipy.stats.skewtest. Useful when you just
                want the "clean" data to be symmetric about the trend.

        Function input allows the user to define a custom test. The function
        must accept two arguments, the trend removed data and a boolean
        array identifying the "clean" data, and return some measure of
        _un_cleanliness (i.e. the result must be smaller if the data are
        "cleaner").

    metric : {string|function}, optional
        Computes tdeviation values for the data. These values are then summed
        to quantify the deviance of each run.

        Built in metrics currently include:
            - 'len' : the number of points in the run
            - 'chi2' : the chi2 statistic for the run assuming unit variance
            - 'area' : area under the curve for the run

        Function input must accept an single array (the detrended) data and
        return a single array of equal length. For example, the 'area'
        function simply returns the absolute value of each point, and these are
        later summed.

    trendfit : {string|function|None}
        Trend to fit and subtract from the partly clean data at each iteration.

        String input can be used to specify one of the following built-in
        trends:
            - 'mean' : mean value of the data
            - 'median' : median value of the data

        Function input allows the user to define a custom trend. The function
        must accept *only* a boolean vector identifying the "clean" data as
        input and return the value of the trend at each data point.

        None means the function will not attempt to fit any trend.

    maxiter : int, optional
        Limits the number of iterations allowed. Error is thrown if this number
        is exceeded.

    Returns
    -------
    clean : 1-D numpy boolean array
        Boolean array identifying the clean (non-anomalous) data points.

    Description
    -----------
    The algorithm consists of three parts:
        (1) Fitting a trend to the data or partially-cleaned data.
        (2) Computing the result of the-user defined test on the detreneded
            data.
        (3) Flagging and removing the most deviant "runs" of data, where
            deviance is quantified by the user-specified metric.

    "Runs" are defined as groups of points succesively above or below the
    trend. The three steps are iterated until the test result is greater than the
    user-specified tolerance, tol and the points identified as clean are the
    same between two iterations.

    The algorithm was designed with the idea
    that it could be used to find and remove flares in time-series observations
    of stars and emission/absorption lines in spectra.

    Modification History
    --------------------
    2014-10-23 written by Parke Loyd
    2015-04-07 heavy modification
    """
    """
    Bad decisions:
        - "Creeping" in or out the edges of the anomalies after each new trendfit:
            the requisite for loops are SLOW. Way faster to just refind the
            anomalies using array operations.
        - iteratively flagging anoms, then refitting and repeating until no
            new points are identified. Since the first fit is high biased due
            to the big anomalies, lots or all of the "good" data is flagged
            and excluded from future fits. better to refit after each anom is
            removed.
    """
    ## PARSE THE INPUT
    n = len(x)

    # construct built in *statisctical* test function (others also built in)
    builtintests = {'runs':runstest, 'r':runstest, 'shapiro-wilks':shapiro,
                    'shapiro':shapiro, 'sw':shapiro, 'skew':skewtest}
    if test in builtintests:
        testfunc = builtintests[test.lower()]
        def test(x, good):
            return 1.0 - testfunc(x[good])[1]

    # construct built in trend fit function
    builtintrends = {'median' : np.median, 'mean':np.mean}
    if type(trendfit) is str:
        fitfunc = builtintrends[trendfit.lower()]
        def trendfit(good):
            value = fitfunc(x[good])
            return np.ones(x.shape)*value

    # construct built in metric
    builtinmetrics = {'chi2' : (lambda x: x**2),
                      'area' : (lambda x: np.abs(x)),
                      'len'  : (lambda x: np.ones(x.shape))}
    if type(metric) is str: metric = builtinmetrics[metric.lower()]

    ## ITERATIVELY FIT AND FLAG ANOMALIES
    Nanom = 0
    if printsteps:
        print 'test value below {} needed to pass'.format(tol)
        print 'anomalies % bad test'
        print '--------- ----- ---------'
    while True:
        Ngood_rec = []
        counter = 0
        good, good_old, Ngood = np.ones(n, bool), np.zeros(n, bool), n
        x1 = x
        while True:
            Ngood_rec.append(Ngood)
            Ngood = np.sum(good)
            counter += 1

            # check various conditions for exiting loop
            if Ngood == 0:
                raise ValueError('All data removed before test was passed.')
            if counter > maxiter:
                raise ValueError('Exceeded allowable number of iterations.')
            if np.all(good_old == good):
                # converged!
                break
            if counter > 10:
                if all(n == Ngood_rec[-10] for n in Ngood_rec[-10::2]):
                    # solution is (almost certainly) oscillating
                    break

            # fit the retained data
            fit = trendfit(good) if trendfit is not None else 0.0
            x1 = x - fit

            # sum deviations over each run of positive or negative points
            deviations, splits = __run_deviations(x1, metric)

            good_old = good
            if Nanom > 0:
                # identify the Nanom largest deviations
                order = np.argsort(np.abs(deviations))
                indices = np.arange(len(deviations))
                sorted_indices = indices[order]
                anomalies = sorted_indices[-Nanom:]

                # flag those anomalies
                good = __buildmask(anomalies, splits, n)
            else:
                good = np.ones(n, bool)

        # check if the remaining data is clean according to the desired
        # condition
        if test in ['devsize', 'deviation size']:
            # maximum deviation that hasn't been removed
            result = np.sort(deviations)[-Nanom - 1]
        elif test in ['sigma clip']:
            # maximum deviation in std dev from mean
            clean_devs = np.sort(deviations)[:(len(deviations)-Nanom)]
            result = np.abs(clean_devs[-1] - np.median(clean_devs)) / np.std(clean_devs)
        else:
            result = test(x1, good)
        if printsteps:
            pctbad = (1.0 - float(Ngood) / n) * 100.0
            print '{:9.0f} {:5.2f} {}'.format(Nanom, pctbad, result)
        if result < tol:
            return good
        else:
            Nanom += 1


def excess_noise(y, base_noise, Poisson=False, PDF=False, CDF=False):
    """Computes the maximum likelihood value of the excess noise.

    Specifically, this function assumes the data, y, are each drawn from a
    normal distribution with the same mean but different variances. The
    variance for the PDF from which a given point was drawn is given by
    base_noise that may differ from point to point and some excess noise that
    is the same for all points.

    Modification History:
    2014/05/12 - Created (only for Poisson=False case)
    """

    y, base_noise = np.array(y), np.array(base_noise)
    ymn = np.mean(y)
    yvar = np.var(y)
    base_var = base_noise**2
    x_guess = np.sqrt(yvar - np.mean(base_var)) if yvar > np.mean(base_var) else 0.0

    def log_like(mn,x_noise,log_norm_fac):
        x_var = x_noise**2
        var = x_var + base_var
        terms = -np.log(2*np.pi*var)/2 - (y - mn)**2/2/var
        return np.sum(terms) + log_norm_fac

    #initial pass at a normalization factor - use peak value
    def neg_like(x):
        mn,x_noise= x
        return -log_like(mn,x_noise,0.0)
    result = _minimize(neg_like, [ymn,x_guess], method='Nelder-Mead')
    log_norm_fac = result.fun
    mn_pk, x_noise_pk = result.x

    def like(mn,x_noise):
        return np.exp(log_like(mn, x_noise, log_norm_fac))

    def xmn_like(xnorm):
        constrained_like = lambda mn: like(mn, xnorm*mn)
        result = _quad(constrained_like, mn_pk, np.inf)[0]
        result += -_quad(constrained_like, mn_pk, 0.0)[0]
        return result

    xmn_pk = x_noise_pk/mn_pk
    result = _minimize(lambda x: -xmn_like(x), xmn_pk, method='Nelder-Mead')
    xmn_pk = result.x[0]
    if xmn_pk < 0.0: xmn_pk = 0.0

    area = _quad(xmn_like, xmn_pk, np.inf)[0]
    area += -_quad(xmn_like, xmn_pk, 0.0)[0]

    if PDF:
        pdf = lambda x: xmn_like(x)/area if x >= 0.0 else 0.0
    if CDF:
        def cdf(x):
            if x <= xmn_pk:
                return -_quad(xmn_like, x, 0.0)[0]/area
            else:
                y = -_quad(xmn_like, xmn_pk, 0.0)[0]/area
                y += _quad(xmn_like, xmn_pk, x)[0]/area
                return y

    if PDF and CDF:
        return xmn_pk, pdf, cdf
    elif PDF:
        return xmn_pk, pdf
    elif CDF:
        return xmn_pk, cdf
    else:
        return xmn_pk


# backward compatible
def excess_noise_PDF(y, base_noise, Poisson=False):
    xmn_pk, pdf = excess_noise(y, base_noise, Poisson=Poisson, PDF=True)
    return pdf, xmn_pk


def poisson_to_skewnorm(rate):
    """
    Approximate a poisson distribution of given rate with a skew normal distribution.

    Parameters
    ----------
    rate
        Rate for Poisson distribution. Must be > 1.0.

    Returns
    -------
    a, loc, scale :
        Parameters for use with scipy.stats.skewnorm. Note that the mean = loc + scale*delta*sqrt(2/pi); var =
        scale**2*(1 - 2*delta**2/pi); skew = (4 - pi)/2 * (delta*sqrt(2/pi))**3/(1 - 2*delta**2/pi)**(3/2); where
        delta = a/sqrt(1 + a**2)
    """
    if rate <= 1.0:
        raise ValueError('Only defined for rate >= 1.0.')

    mean, var, skew = poisson.stats(rate, moments='mvs')

    delta = np.sqrt(np.pi/2*skew**(2./3)/(skew**(2./3) + ((4-np.pi)/2)**(2./3)))
    a = delta/np.sqrt(1 - delta**2)
    scale = np.sqrt(var/(1 - 2*delta**2/np.pi))
    loc = mean - scale*delta*np.sqrt(2/np.pi)
    return a, loc, scale


def __run_deviations(x, metric):
    pospts = (x > 0)
    splits = np.nonzero(pospts[1:] - pospts[:-1])[0] + 1
    ptdevs = metric(x)
    deviations = splitsum(ptdevs, splits)
    return deviations, splits

def __buildmask(flags, splits, n):
    # split a vector of all True values up by run
    alltrue = np.ones(n, bool)
    runs = np.array(np.split(alltrue, splits))
    # then mark the flags as false
    runs[flags] = runs[flags]*False
    # and reconstruct the vector
    good = np.hstack(runs)
    return good
