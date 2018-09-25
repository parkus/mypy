# -*- coding: utf-8 -*-
"""
Created on Wed May 21 14:10:49 2014

@author: Parke
"""
import numpy as np
import my_numpy as mnp
from scipy.integrate import quad
import scipy.special as ss
import math

__all__ = ['optimal_grid','confidence_interval','gauss_findpt']


def inv_gauss_cdf(prob):
    """Find the value where a unit gaussian CDF has *two-tailed* probability prob."""
    return abs(math.sqrt(2) * ss.erfinv(prob - 1))


def gauss_integrate_tail(pdf, a, b, sigma_guess=1.0):
    """Rapidly integrates the tail of a gaussian by assuming that it can be
    approximated as the function f(x) = A*exp(-B*x**2)
    
    sigma_guess is used to select two points to sample the tail to compute A
    and B. 
    
    The absolute error is roughly pdf(pt nearest peak)/pdf(peak)/100.
    """
#    pdb.set_trace()
    pa, pb = pdf(a), pdf(b)
    if pa > pb: 
        x1, p1 =  a, pa
        x2 = a - sigma_guess if a > b else a + sigma_guess
    else:
        x1, p1 =  b, pb
        x2 = b - sigma_guess if b > a else b - sigma_guess
    p2 = pdf(x2)
    # assume the tail has the form f = A*exp(-B*x^2)
#    if x1 == x2: pdb.set_trace()
    B = np.log(p2/p1)/(x1**2 - x2**2)
    A = p1*np.exp(B*x1**2)
    rootB = np.sqrt(B)
    return A/2*np.sqrt(np.pi)/rootB*(ss.erf(rootB*b) - ss.erf(rootB*a))

def gauss_integrate(pdf, mu, a, b, sigma_guess=1.0, epsabs=1.49e-8):
    """NOT FULLY WRITTEN. Use the fact that most pdfs are gaussian-like to 
    rapidly and accurately integrate.
    
    Uses quad near the peak, then uses erf for the tail(s). pdf is a funtion
    reference.
    """
    pmax, pa, pb = pdf(mu), pdf(a), pdf(b)
    pbreak = pmax*10.0*epsabs #chose the "start" of the tail to get roughly the right tolerance
    if pa >= pbreak and pb >= pbreak:
        I = quad(pdf, a, b, epsabs=epsabs)[0]
    elif pa < pbreak and pb >= pbreak:
        guess = mu-sigma_guess if a < mu else mu+sigma_guess
        xbreak = gauss_findpt(pdf, mu, pbreak, guess)
        I = gauss_integrate_tail(pdf, a, xbreak, sigma_guess) + \
            quad(pdf, xbreak, b, epsabs=epsabs)[0]
    elif pa >= pbreak and pb < pbreak:
        guess = mu-sigma_guess if b < mu else mu+sigma_guess
        xbreak = gauss_findpt(pdf, mu, pbreak, guess)
        I = quad(pdf, a, xbreak, epsabs=epsabs)[0] + \
            gauss_integrate_tail(pdf, xbreak, b, sigma_guess)
    else: #pa < pbreak and pb < pbreak
        I = gauss_integrate_tail(pdf, a, b, sigma_guess)
    return I

def upper_limit(pdf,confidence=0.95,normalized=False,x0=-np.inf,xpeak=None,
                x1guess=None):
    """Find the endpoint enclosing total probability prob for a pdf.
    
    Input pdf can be a list [x,p] containing x and p arrays sampling the pdf or
    a function pdf that gives the probability density at x. xpeak is the point
    at which the peak of the pdf occurs (use scipy.optimize.minimize), useful
    (but optional) only when pdf is a function to ensure numerical integration
    using quad does not miss a narrow peak.
    
    Haven't yet implemented the function version of this... see comments in
    code for thoughts on how to do so.
    """

    if hasattr(pdf, '__call__'): #see if it is a function
        tol = 10**(-len(str(confidence))+1)
        if not normalized:
            if xpeak: total = (-quad(pdf, xpeak, x0)[0] + 
                               quad(pdf, xpeak, np.inf)[0])
            else: total = quad(pdf, x0, np.inf)[0]
            pdf = lambda x: pdf(x)/total
        if not x1guess:
            x1guess = xpeak*2.0 if xpeak else 1.0
        
        x1old = x0
        Iold = 0.0
        x1 = x1guess
        if xpeak:
            I = (-quad(pdf, xpeak, x1old, epsabs=tol)[0] + 
                 quad(pdf, xpeak, x1, epsabs=tol)[0])
        else:
            I = quad(pdf, x1old, x1, epsabs=tol)[0]
        while abs(I - Iold) > tol:
            dIdx = (I - Iold)/(x1 - x1old)
            x1old = x1
            dx1 = (confidence - I)/dIdx
            x1 += dx1
            Iold = I
            dI = quad(pdf, x1old, x1, epsabs=tol)[0]
            I += dI
        return x1
        
    elif type(pdf) is list:
        x,p = np.array(pdf[0]), np.array(pdf[1])
        if not normalized: p = p/np.trapz(p,x)
        total = 0.0
        i = 0
        while total < confidence:
            total += (p[i] + p[i+1])/2.0*(x[i+1] - x[i])
            i += 1
        
        x = (x[i-1] + x[i])/2.0 #could make this better, but...
        return x
        
    else:
        print 'Bad input, see docstring.'
        return

def confidence_interval(pdf, xpeak=None, confidence=0.683, normalized=False, return_xpeak=False, use_mean=False):
    """Find the endpoints enclosing a certain total probability prob for a pdf.    
    
    Input pdf can be a list [x,p] containing x and p arrays sampling the pdf, a list [xedges, p] for a histogram of
    samples of the pdf (i.e. histogrammed MCMC output), or
    a function pdf that gives the probability density at x. xpeak is the point
    at which the peak of the pdf occurs (use scipy.optimize.minimize), required
    only when pdf is a function.
    
    Note to self: if the pdf should be zero outside of some interval (e.g. if 
    you have applied a bayesian prior), make sure the input function returns
    zero outisde of that interval.
    
    The algorithm for when pdf is a function could be dramatically sped up by
    using the result of previous integrations to save time in computing future
    integrations. I think, anyway.

    Parameters
    ----------
    use_mean :
        If True, use mean rather than mode of pdf for xpeak.
    """
    

    
    if hasattr(pdf, '__call__'): #see if it is a function
        #set tolerance to one digit better precision than the specified confidence
        tol = 10**(-len(str(confidence))+1)
        xtol = tol*xpeak
        Itol = 10*xtol

        pmax = pdf(xpeak)
        pguess = pmax*(1 - confidence)
        ptol = pmax*tol
        
        if not normalized:
            total = (-quad(pdf, xpeak, -np.inf)[0] + quad(pdf, xpeak, np.inf)[0])
            _pdf = pdf
            pdf = lambda x: _pdf(x)/total
        
        #use the fact that the pdf is normalized to guess at its width, then
        #use that to guess at the endpts
        width = 1.0/pmax
        x1 = xpeak + width/2.0
        x0 = xpeak - width/2.0
        
        #now use newton-raphson to find the p value that gives the right integral
        pold = pmax
        p = pguess
        Iold, I = 0.0, 0.0
        x0 = gauss_findpt(pdf, xpeak, p, x0, abstolx=xtol)
        x1 = gauss_findpt(pdf, xpeak, p, x1, abstolx=xtol) 
        x0old, x1old = xpeak, xpeak
        while abs(p - pold) > ptol:
            dI = (-quad(pdf, x0old, x0, epsrel=Itol)[0]
                  +quad(pdf, x1old, x1, epsrel=Itol)[0])
            I += dI
            dIdp = (I - Iold)/(p - pold)
            pold = p
            p += (confidence - I)/dIdp
            Iold = I
            x0old, x1old = x0, x1
            x0 = gauss_findpt(pdf, xpeak, p, x0old, abstolx=xtol)
            x1 = gauss_findpt(pdf, xpeak, p, x1old, abstolx=xtol)
            
    elif type(pdf) is list:

        x,p = np.array(pdf[0]), np.array(pdf[1])

        if len(x) == len(p):
            if not normalized: p = p/np.trapz(p,x)
            imax = np.argmax(p)
            xpeak = x[imax]
            I = mnp.cumtrapz(p, x, zero_start=True)
        elif len(x) == len(p) + 1:
            dx = np.diff(x)
            xmids = mnp.midpts(x)
            xpeak = xmids[np.argmax(p)]
            areas = dx*p
            I = np.insert(np.cumsum(areas), 0, 0)
        if use_mean:
            x, x0, x1 = _cdf_endpoints(x, I, confidence, 'mean')
        else:
            x0, x1 = _cdf_endpoints(x, I, confidence, xpeak)

    else:
        print 'Bad input -- read the docstring.'
        return

    if return_xpeak:
        return xpeak, x0, x1
    else:
        return x0, x1
            

def _cdf_endpoints(x, I, confidence, xpeak):
    I = I/I[-1]
    if xpeak == 'mean':
        Ipk = 0.5
        xpk = np.interp(Ipk, I, x)
    else:
        Ipk = np.interp(xpeak, x, I)
    I0, I1 = Ipk - confidence/2.0, Ipk + confidence/2.0
    if I0 < 0 or I1 > 1:
        raise ValueError('PDF is highly asymmetric. Less than confidence/2. area of the PDF is to one or both sides '
                         'of the peak, so normal confidence interval is ill-defined.')
    x0, x1 = np.interp([I0, I1], I, x)
    if xpeak == 'mean':
        return xpeak, x0, x1
    else:
        return x0, x1


def gauss_findpt(fun, mu, fval, guess, abstolx=1e-6, maxiter=1e3):
    """Search for the point x where fun(x) = fval, assuming that fun is
    Gaussian-like with mean mu. Choose guess such that the result is on the
    side of the Gaussian that you desire.
    
    Assumes you know mu and treats this as a fixed parameter. Let's hope that
    works bc otherwise the math gets messy. It seems to work based on
    supplying f = exp(-2*x**2) - 0.2*x**2.
    """
    sign = 1 if guess > mu else -1
    xold = mu
    fold = fun(mu)
    xnew = guess
    it = 0
    while abs(xnew - xold) > abs(abstolx) and it < maxiter:
        fnew = fun(xnew)
        B = np.log(fnew/fold)/((xnew - mu)**2 - (xold - mu)**2)
        A = fnew/np.exp(B*(xnew - mu)**2)
        xold = xnew
        fold = fnew
        xnew = mu + sign*np.sqrt(np.log(fval/A)/B)
        it += 1
    return xnew


def optimal_grid(fun,x_max,Npts,to_level=(1e-5,1e-5),bounds=(-np.inf,np.inf)):
    """Generates a grid of points to optimally sample a function with a single
    peak at x_max. Created to sample PDFs.
    
    Note that this function will evaluate more points than requested. The idea 
    is to appropriately downsample the grid for multidemnsional pdfs -- e.g.
    use 1000 pts to figure out how to best space 100 pts for eventual use
    in a 100x100x100 grid.
    
    Npts will be made odd if it is not already so that an even number of points
    can be on each side of a peak point.
    
    Could do this better by looking at the difference of the integrals, but
    this does okay for now. Currently it oversamples regions where the function
    value is small but the curvature is still high.
    """
    bounds = [float(b) for b in bounds]
    f_max = fun(x_max)
    if Npts % 2 == 0: Npts += 1
      
    #find x value that reaches level to both sides of x_start
     
    def find_x(bound,guess,fval):
        if fun(bound) >= fval:
            return bound        
        else:
            return gauss_findpt(fun, x_max, fval, guess, 
                                abstol=abs(guess - x_max)/Npts/10.0)
    
    guess_left = (x_max - bounds[0])/2.0 if np.isfinite(bounds[0]) else x_max - 1
    guess_right = (bounds[1] - x_max)/2.0 if np.isfinite(bounds[1]) else x_max + 1
    while fun(guess_left) <= 0.0:
        guess_left = guess_left + (x_max - guess_left)/2.0
    while guess_right <= 0.0:
        guess_right = guess_right + (guess_right - x_max)/2.0
    x_left = find_x(bounds[0], guess_left, f_max*to_level[0])
    x_right = find_x(bounds[1], guess_right, f_max*to_level[1])
    
    #finely sample the function between the bounds
    dx = (x_right - x_left)/(Npts*10)
    x_smpl2 = np.arange(x_left-2*dx, x_right+2.5*dx, dx)
    x_smpl = x_smpl2[2:-2]
    
    #estimate slope at each pt
    f_smpl = np.array([fun(x) for x in x_smpl2])
    slopes = (f_smpl[2:] - f_smpl[:-2])/2/dx
    curves = abs(slopes[2:] - slopes[:-2])/2/dx
    
    #it is tempting to use interpolation instead of this cloogefest, but note
    #that curve is not a single valued functino of x, so this is difficult
    def chunkify(smplvec,chunkvec,chunksize):
        partsum = 0
        chunk_endpts = []
        for i,cpt in enumerate(chunkvec):
            partsum += cpt
            if partsum > chunksize:
                chunk_endpts.append(i)
                partsum -= chunksize
        return smplvec[np.array(chunk_endpts)] 
    
    pkpt = np.argmax(f_smpl) - 2
    Sleft = sum(curves[:pkpt])
    Sright = sum(curves[pkpt+1:])
    S = Sleft + Sright
    Nchunks = Npts - 1
    ratio_left = Sleft/S
    Nchunks_left = round(ratio_left*Nchunks)
    Nchunks_right = Nchunks - Nchunks_left
    spacing_left = Sleft/Nchunks_left
    spacing_right = Sright/Nchunks_right
    
    x_grid_left = chunkify(x_smpl[pkpt-1::-1],curves[pkpt-1::-1],spacing_left)
    x_grid_right = chunkify(x_smpl[pkpt+1:],curves[pkpt+1:],spacing_right) 
    x_grid_left = x_grid_left[::-1]
    x_grid = ([x_left] + list(x_grid_left) + [x_max] + list(x_grid_right) + 
              [x_right])
    return np.array(x_grid)


def mode_halfsample(x, presorted=False):
    """
    Estimate the mode from which x was drawn using the halfsample method (HSM).

    Parameters
    ----------
    x : array-like
        The randomly sampled data for which to find the mode.
    presorted : boolean
        True if the data are already sorted (can save time if function is looped). Default is False.

    Returns
    -------
    mode : float
        Estimated mode of the distribution from which x was drawn.

    Notes
    -----
    Sometimes other ways of estimating the mode perform better, such as kernel density estimation, for specific
    forms of the pdf. However, this method is accurate, fairly robust, and very fast.

    Examples
    --------
    x = np.random.exponential(size=100000)
    mode_halfsample(x)
    # should return a value near 0

    References
    ----------
    Bickel & Fruhwirth 2005 https://arxiv.org/abs/math/0505419
    Robertson & Cryer 1974 Journal of the American Statistical Association 69:1012

    """
    if not presorted:
        x = np.sort(x)
    n = len(x)
    while n > 2:
        # roughly half the number of samples
        h = int(math.ceil(len(x)/2))

        # find the half-sample with the shortest range
        ranges = x[h:] - x[:-h]
        i = np.argmin(ranges)

        # make that the new data vector and do the whole thing over again
        x = x[i:i+h]
        n = len(x)
    return np.mean(x)


class OutOfRange(Exception):
    pass


def mcmc_error_bars(x, x_ml=None, interval=0.683, limit=0.95, method='percentile'):
    """
    Find the most likely value and error bars (confidence interval) of a parameter based on random samples (as from
    an MCMC parameter search). If a most likely value with a confidence interval is not well defined, return an upper
    or lower limit as appropriate.

    Parameters
    ----------
    x : array-like
        The randomly sampled data for which to find a most likely value and error bars.
    x_ml : float
        Max-likelihood value of x. Useful when this value was found with, say, scipy.optimize.minimize and now you
        just want to get the error bars to either side of that value, even though the MCMC sampling might show a
        slightly different peak.
    interval : float
        The width of the confidence interval (such as 0.683 for 1-sigma error bars).
    limit : float
        The cumulative probability at which to set an upper or lower limit as necessary.

    Returns
    -------
    x_mode, err_neg, err_pos : floats
        The most probable x value and the negative (as a negative value) and positive error bars. If searching for a
        confidence interval resulted in a limit, then np.nan is used for x_mode and err_neg (upper limit) or err_pos
        (lower limit) and the limit value is given in the remaining position.

    Examples
    --------
    import numpy as np
    from matplotlib import pyplot as plt
    x = np.random.normal(10., 2., size=100000)
    _ = plt.hist(x, 200) # see what the PDF it looks like
    error_bars(x)
    # should return roughly 10, -2, 2

    x = np.random.gamma(1.0, 2.0, size=100000)
    _ = plt.hist(x, 200) # see what the PDF looks like
    error_bars(x)
    # should return an upper limit of roughly 6 as nan, nan, 6

    Notes
    -----
    The confidence interval is taken to be the central chunk of the PDF corresponding to the value set by `interval`.
    Error bars are then defined by the difference from the lower and upper limits of this chunk to the max-likelihood
    value. If the max-likelihood value is outside of these limits, then the PDF is interpreted as only giving an upper
    or lower limit as appropriate, and that limiting value is computed and returned instead.

    """

    if np.any(np.isnan(x)):
        raise ValueError('There are NaNs in the samples.')

    if method == 'percentile':
        x50 = np.percentile(x, 50)
        dp = interval/2*100
        return [x50] + (np.percentile(x, [50 - dp, 50 + dp]) - x50).tolist()
    elif method == 'maxlike':
        x = np.sort(x)
        if x_ml is None:
            x_ml = mode_halfsample(x, presorted=True)

        interval_pcntl = 100 * np.array([(1-interval)/2, (1+interval)/2])
        x_min, x_max = np.percentile(x, interval_pcntl)

        if x_ml < x_min: # upper limit
            return [np.nan, np.nan, np.percentile(x, 100*limit)]

        if x_ml > x_max: # lower limit
            return [np.nan, np.percentile(x, 100*(1-limit)), np.nan]

        # interval is good
        return x_ml, x_min - x_ml, x_max - x_ml
    else:
        raise ValueError('Method can only be either "percentile" or "maxlike".')