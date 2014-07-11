def detect_correlations(tx,x,ty,y,sample_dt,Ntrials=1e4):
    import numpy as np
    from scipy import interpolate
    
    #validate input
    if len(x) != len(tx):
        print "The x and tx vectors should be the same length."
        return
    if len(y) != len(ty):
        print "The y and ty vectors should be the same length."
        return
    
    # mean-subtract and sigma-normalize the data vectors
    x = (x - np.mean(x))/np.std(x)
    y = (y - np.mean(y))/np.std(y)
    
    # first,initialize some variables
    significance = np.zeros(len(sample_dt))
    correlations = np.zeros(len(sample_dt))
    
    # generate simulted data to test null hypothesis of white noise
    simx = np.random.normal(0,1,[Ntrials,len(x)])
    simy = np.random.normal(0,1,[Ntrials,len(y)])
    
    # make vectors of indices for later use
    xindices = np.arange(len(x))
    yindices = np.arange(len(y))
    
    # --------------------------------------------------------------------------    
    # define function for computing correlation and significance
    def get_sig(x,y,xsim,ysim):
        corr = np.sum(x*y)
        simcorr = np.sum(xsim*ysim,1)
        sig = float(sum(abs(simcorr) > abs(corr)))/np.size(xsim, 0)
        return (corr,sig)
    
    # --------------------------------------------------------------------------
    # loop through possible dts
    for i, dt in enumerate(sample_dt):
        ty_shifted = ty - dt
        
        #figure out the range over which tx and ty-dt overlap and which vector
        #contains the first or last point in the overlapping interval
        #starting indices
        if tx[0] >= ty_shifted[0]:
            firstpt = 'x'
            x0 = 0
            y0 = min(yindices[ty_shifted >= tx[0]])
        else:
            firstpt = 'y'
            x0 = min(xindices[tx > ty_shifted[0]])
            y0 = 0
        #ending indices
        if tx[-1] <= ty_shifted[-1]:
            lastpt = 'x'
            x1 = len(tx)
            y1 = max(yindices[ty_shifted <= tx[-1]])+1
        else:
            lastpt = 'y'
            x1 = max(xindices[tx < ty_shifted[-1]])+1
            y1 = len(ty_shifted)
        overlap = x1-x0
        
        #-----------------------------------------------------------------------
        #interpolate and compute significance
        
        #if the data are exactly aligned, no need to interpolate
        if (x1-x0 == y1-y0) and sum(tx[x0:x1] == ty_shifted[y0:y1]).all():
            result = get_sig(x[x0:x1],y[y0:y1],simx[:,x0:x1],simy[:,y0:y1])
            correlations[i] = result[0]/overlap
            significance[i] = result[1]
        else: #try interpolating y onto x and x onto y. choose the better
            #first y onto x
            y0i, y1i = y0, y1
            if firstpt == 'x': y0i = y0 - 1
            if lastpt == 'x': y1i = y1 + 1
            #interpolate real data
            interp = interpolate.interp1d(ty_shifted[y0i:y1i], y[y0i:y1i])
            yinx = interp(tx[x0:x1])
            #interpolate simulated data
            interpsim = interpolate.interp1d(ty_shifted[y0i:y1i],simy[:,y0i:y1i])
            simyinx = interpsim(tx[x0:x1])
            result = get_sig(x[x0:x1],yinx,simx[:,x0:x1],simyinx)
            corr_y2x = result[0]/overlap
            significance_y2x = result[1]
            
            #now x onto y
            x0i, x1i = x0, x1
            if firstpt == 'y': x0i = x0 - 1
            if lastpt == 'y': x1i = x1 + 1
            #interpolate real data
            interp = interpolate.interp1d(tx[x0i:x1i], x[x0i:x1i])
            xiny = interp(ty_shifted[y0:y1])
            #interpolate simulated data
            interpsim = interpolate.interp1d(tx[x0i:x1i],simx[:,x0i:x1i])
            simxiny = interpsim(ty_shifted[y0:y1])
            result = get_sig(xiny,y[y0:y1],simxiny,simy[:,y0:y1])
            corr_x2y = result[0]/overlap
            significance_x2y = result[1]
            
            #choose the better
            if significance_y2x < significance_x2y:
                significance[i] = significance_y2x
                correlations[i] = corr_y2x
            else:
                significance[i] = significance_x2y
                correlations[i] = corr_x2y            
        
    return (correlations,significance)