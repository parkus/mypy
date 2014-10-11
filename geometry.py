# -*- coding: utf-8 -*-
"""
Created on Sun Sep 07 13:39:08 2014

@author: Parke
"""
import numpy as np
import math

def dist(x0,y0,x1,y1):
    return np.sqrt((x1-x0)**2 + (y1-y0)**2)
    
def polygon_area(*args):
    """Compute the area of a polygon given a set of vertices provides either
    as sperate x,y vectors or an Nx2 or 2xN array of (x,y) pairs.
    
    Points in counterclockwise order give positive area. CW order gives negative.""" 
    if len(args) == 1:
        verts = np.array(args)
        if verts.shape[0] == 2: verts = verts.T
        x = verts[0]
        y = verts[1]
    if len(args) == 2:
        x,y = map(np.array, args)
    return 0.5*(np.sum(x[:-1]*y[1:]) - np.sum(x[1:]*y[:-1]))

def brute_area(xrange,yrange,in_or_out,Nsamples=1e4):
    dx, dy = xrange[1] - xrange[0], yrange[1] - yrange[0]
    area = dx*dy
    xvec = np.random.uniform(xrange[0],xrange[1],Nsamples)
    yvec = np.random.uniform(yrange[0],yrange[1],Nsamples)
    
    try:
        cnt = np.sum(in_or_out(xvec,yvec))
    except:
        cnt = 0
        for x,y in zip(xvec,yvec):
            cnt += in_or_out(x,y)
        
    return area*float(cnt)/Nsamples

def circle_area_segment(*args):
    """Compute the area of the segment of a circle given the radius and center-
    to-chord distance or the radius and the x0,y0,x1,y1 coordinates of the chord
    endpoints (useful for finding areas of circle-circle intersections when
    the intersection points are known)."""
    if len(args) == 1:
        args = args[0]
    if len(args) == 5:
        r = args[0]
        chordlen = dist(*args[1:])
        sectionangle = 2.0*math.asin(chordlen/2.0/r)
        sectionarea = sectionangle/2.0*r**2
        triheight = math.sqrt(r**2 - (chordlen/2.0)**2)
        triarea = 0.5*triheight*chordlen
        segmentarea = sectionarea - triarea
    elif len(args) == 2:
        r,triheight = args
        sectionangle = 2*math.acos(triheight/r)
        sectionarea = sectionangle/2.0*r**2
        chordlen = 2*math.sqrt(r**2 - triheight**2)
        triarea = 0.5*triheight*chordlen
        segmentarea = sectionarea - triarea
    else:
        raise ValueError('Not a valid number of arguments. See function.')
    return segmentarea

def circle_circle_pts(*args):
    """Computes the points where two circles interesect.
    
    If the circles do not interset, an empty list is returned. If one circle is
    enclosed in another, the index ([0] or [1]) of that circle is returned.
    Otherwise, the intersection points are returned as the tuple (xi0,yi0),
    (xi1,yi1) with xi0 >= xi1.
    """
    if len(args) == 2:
        [x0,y0,r0],[x1,y1,r1] = args
    if len(args) == 6:
        x0,y0,r0,x1,y1,r1 = args
        
    d = dist(x0,y0,x1,y1)
    if d >= (r0+r1):
        return []
    elif d < abs(r1-r0):
        return [1] if r1 < r0 else [0]
    else:
        #compute intersection
        q = (r0**2 - r1**2 + x1**2 - x0**2 + y1**2 - y0**2)/2.0
        dx, dy = (x1 - x0), (y1 - y0)
        a = 1 + dx**2/dy**2
        b = -2*x0 - 2*q*dx/dy**2 + 2*y0*dx/dy
        c = x0**2 + y0**2 + q**2/dy**2 - 2*q*y0/dy - r0**2
        xi0 = (-b + math.sqrt(b**2 - 4*a*c))/2/a
        xi1 = (-b - math.sqrt(b**2 - 4*a*c))/2/a
        yi0, yi1 = (q - xi0*dx)/dy, (q - xi1*dx)/dy
        return np.array([[xi0,yi0], [xi1,yi1]])

def circle_circle_area(circles,kind='union',intersectpts=None):
    """Find the area of the union, intersection, or difference of two circles.
    """
    if intersectpts == None:
        intersectpts = circle_circle_pts(circles)
    cAreas = __cArea(circles)
    if intersectpts == []:
        return 0.0
    else:
        seginput = [np.hstack([c[-1],intersectpts.ravel()]) for c in circles]
        segAreas = map(circle_area_segment, seginput)
    oppAreas = cAreas - segAreas
    
    #check whether the circle centers lie to the same side of the chord. This
    #will be true if the dot product of the chord-center-circle-center vectors
    #is positive
    chordcen = np.sum(intersectpts,0)/2.0
    cenvecs = circles[:,:2] - chordcen[np.newaxis,:]
    sameside = (np.dot(*cenvecs) > 0)
    
    if kind.lower() in ['union', 'u']:
        area = max(oppAreas) + max(segAreas) if sameside else sum(oppAreas)
    if kind.lower() in ['intersection', 'x']:
        area = min(oppAreas) + min(segAreas) if sameside else sum(segAreas)
    if kind.lower() in ['difference', 'd']:
        if sameside:
            area = max(oppAreas) + max(segAreas) - min(oppAreas) - min(segAreas)
        else:
            area = sum(oppAreas) - sum(segAreas)
    return area
        
def circles_area_union(circles, intersections=None, brutegrid=False):
    """Compute the area of the union of a set of circles, input as an Nx3 array
    or the list equivalent. Brutegrid uses a the random grid method (good for
    checkin and might be faster for large numbers of circles).
    """
    circles = np.array(circles)
    
    if brutegrid:
        xc,yc,r = circles.T
        def in_circles(x,y):
            d = dist(x,y,xc,yc)
            return any(d < r)
        
        xrange = [np.min(xc-r), np.max(xc+r)]
        yrange = [np.min(yc-r), np.max(yc+r)]
        
        return brute_area(xrange, yrange, in_circles, brutegrid)
        
    def onborder(pt):
        xi,yi,c0,c1 = pt
        others = circles[[i for i in range(len(circles)) if i not in [c0,c1]]]
        d = dist(xi,yi,others[:,0],others[:,1])
        return not any(d < others[:,2])
    
    #get intersection points and keep only those on borders
    if intersections == None: intersections = circle_intersection_pts(circles)
    xpts,xflags,incircs = intersections
    xpts = np.array(filter(onborder, xpts))
    
    #compute group and pair areas
    grouparea = __circle_group_areas(circles,xpts,'union')
    
    inflags = np.array([len(inc) > 0 for inc in incircs], bool)
    loners = np.logical_not(np.logical_or(inflags,xflags))
    lonerareas = list(__cArea(circles[loners]))
    
    return grouparea + sum(lonerareas)
        
def circleset_area_intersect(circles0, circles1, intersections=None,
                                brutegrid=False):
    circles = np.vstack([circles0,circles1])
    N = len(circles)
    divider = len(circles0)
    
    if brutegrid:
        xc,yc,r = circles.T
        def in_intersection(x,y):
            d = dist(x,y,xc,yc)
            incircs = d < r
            return any(incircs[:divider]) and any(incircs[divider:])
        
        xrange = [np.min(xc-r), np.max(xc+r)]
        yrange = [np.min(yc-r), np.max(yc+r)]
        
        return brute_area(xrange, yrange, in_intersection, brutegrid)
    
    #get intersection points
    if intersections == None: intersections = circle_intersection_pts(circles)
    xpts,xflags,incircs = intersections
    
    #filter out the points we want from all of the
    #identify loners at the same time (since loners can have intersections, but
    #only if they are solely with the opposing set)
    global n, xptsnew
    xptsnew = np.zeros(xpts.shape)
    n = 0 #counter
    xselfflags = np.zeros(N, bool) #track whether a circle ever intersects its own set
    def keepit(pt):
            global n, xptsnew
            xptsnew[n] = pt
            n += 1
    
    for pt in xpts:
        xi,yi,c0,c1 = pt
        
        #figure out which circles the point is in
        others = circles[[i for i in range(len(circles)) if i not in [c0,c1]]]
        d = dist(xi,yi,others[:,0],others[:,1])
        newdiv = divider - int(c0 < divider) - int(c1 < divider)
        inside = d < others[:,2]
        inset0 = any(inside[:newdiv])
        inset1 = any(inside[newdiv:])
        
        #keep only those between circles of different types if they are within 
        #only 0 or 1 sets
        if (c0 < divider and c1 >= divider) or (c0 >= divider and c1 < divider):
            if inset0 + inset1 < 2:
                keepit(pt)
        #if the pt is between two circles of the same set
        else:
            if c0 < divider and not inset0 and inset1:
                keepit(pt)
            if c0 >= divider and not inset1 and inset0:
                keepit(pt)
            xselfflags[[c0,c1]] = True
    xpts = xptsnew[:n]
    
    #identify lone circles fully within just the opposing set
    inotherset = np.zeros(N, bool)
    for i,inc in enumerate(incircs):
        if len(inc) == 0: continue
        inc = np.array(inc)
        if i < divider:
            if any(inc >= divider) and (not any(inc < divider)): 
                inotherset[i] = True
        else:
            if any(inc < divider) and (not any(inc >= divider)):
                inotherset[i] = True
    #and that do not intersect other circles in their set
    loners = np.logical_and(np.logical_not(xselfflags), inotherset)
    lonerAreas = __cArea(circles[loners]) #compute their area
    
    #discard any xpts on loner circles
    keep = np.ones(n, bool)
    for c in np.nonzero(loners)[0]:
        keep = np.logical_and(xpts[:,2] != c, xpts[:,3] != c)
        xpts = xpts[keep]        
    
    #compute group and pair areas
    groupareas = __circle_group_areas(circles,xpts,'intersection',divider)

    return np.sum(lonerAreas) + np.sum(groupareas)
    
def circleset_area_difference(circles0,circles1,intersections=None,
                              brutegrid=False):
    pass
    
def circleset_area_subtract(circles, subcircles, intersections=None, 
                            brutegrid=False):
    """Compute the area of the union of circles and subtract any area that
    overlaps with the circles in subcircles.
    
    Circles are Nx3 arrays of x,y,r or the list equivalent. 
    
    Note that this is not the same as computing area of the difference of two
    shapes, since that would include area in the subtracted circles that does
    not overlap with the original circles.
    """
    if brutegrid:
        pass
    
    allcircles = np.vstack([circles,subcircles])
    divider = len(circles)
    if intersections == None: xpts = circle_intersection_pts(allcircles)
    
    onlyin0 = np.logical_and(xpts[:,2] < divider, xpts[:,3] < divider)
    onlyin1 = np.logical_and(xpts[:,2] >= divider, xpts[:,3] >= divider)
    xpts0, xpts1 = xpts[onlyin0], xpts[onlyin1]
    Asets = [circles_area_union(c,x) for c,x in [[circles, xpts0], 
                                                [subcircles, xpts1]]]
    Aint = circleset_area_intersect(circles,subcircles,xpts)
    return Asets[0] - Aint

def __circpolyarea(circles,polygon,kind='union',divider=None):
    n = len(polygon)
    N = len(circles)    
    if n <= 3:
        raise ValueError('Polygon with 3 ot fewer vertices encountered.')
    x,y = polygon.T[:2]
    polyarea = polygon_area(x,y)
    borderareas = np.zeros(len(polygon)-1)
    
    def checkline(pt0,pt1,c):
        i = np.arange(divider,N) if c < divider else np.arange(0,divider)
        return __line_in_a_circle(pt0,pt1,circles[i])
    
    for j in range(n-1):
        x0,y0,c00,c01 = polygon[j]
        x1,y1,c10,c11 = polygon[j+1]
        chord = np.array([x1-x0, y1-y0])
        #if the points are on the same two circles
        if (c00 == c10 and c01 == c11) or (c00 == c11 and c01 == c10):
            #check if c00 circle center is right of the cord (outside the
            #polygon). if so, use it. else, use c01. 
            cens = circles[[c00,c01],:2]
            xpt0 = np.array([x0,y0])
            nadirs = cens - xpt0[np.newaxis,:]
            outside = __CCWmeasure(chord,nadirs) > 0.0
            if kind == 'union':
                c = c00 if outside[0] else c01
            elif kind == 'intersection':
                #use whichever segment sticks out more but is still entirely
                #within another circle
                chordlen = dist(x0,y0,x1,y1)
                radii = circles[[c00,c01],2]
                segheights = radii - np.sqrt(radii**2 - chordlen**2/4.0)
                if outside[0]: segheights[0] = 2*radii[0] - segheights[0]
                if outside[1]: segheights[1] = 2*radii[1] - segheights[1]
                if x1 == x0:
                    dx, dy = segheights, 0.0
                else:
                    chordslope = (y1-y0)/(x1-x0)
                    dy = np.sqrt(segheights**2/(1 + chordslope**2))
                    dx = abs(chordslope*dy)
                if x1 > x0: dy = -dy
                if y1 < y0: dx = -dx
                chordcen = np.array([x0+x1, y0+y1])/2.0
                farpoints = chordcen[np.newaxis,:] + np.vstack([dx,dy]).T
                incircs = [checkline(chordcen,fp,c) for fp,c in 
                           zip(farpoints,[c00,c01])]
                if all(incircs):
                    #if segments are both within another circle, pick the largest
                    bigger = np.argmax(segheights)
                    c = [c00,c01][bigger]
                else:
                    #pick whichever is in a circle
                    c = c00 if incircs[0] else c01
        else:
            c = c00 if c00 in [c10,c11] else c01
        xc,yc,r = circles[c]
        segarea = circle_area_segment(r,x0,y0,x1,y1)
        nadir = np.array([[xc-x0, yc-y0]])
        inside = (__CCWmeasure(chord,nadir)[0] < 0.0)
        borderareas[j] = segarea if inside else __cArea(circles[[c]]) - segarea
    return polyarea + np.sum(borderareas)
    
def __circle_group_areas(circles,xpts,kind='union',divider=None):
    """Compute the area of groups of intersecting circles by identifying the
    polygons constructed from xpts, computing the polygon area, and adding
    the area of the bordering circle segments.
    
    only kind = 'union' and 'intersection' are supported. If 'intersection' is
    specified, the divider keyword must also be supplied to let the function
    know the slice index for circles that divides the circles of the first and
    second sets.
    """
    def getconnector(c0,c1,c):
        if c0 in c:
            if c1 in c: return [c0,c1]
            return [c0]
        if c1 in c: return [c1]
        return []
        
    #find nearest counterclockwise intersection pt on circcle c
    def nextpt(pt,pts):
        c0,c1 = pt[2:]
        cs = pts[:,2:]
        connectors = [getconnector(c0,c1,c) for c in cs]
        connected = np.array([len(c) for c in connectors], bool)
        if sum(connected) == 0:
            raise ValueError('Well, that shouldn\'t have happened. Couldn\'t find a next point.')
        elif sum(connected) == 1:
            return np.nonzero(connected)[0][0]
        if sum(connected) > 1:
            chords = pts[connected,:2] - pt[np.newaxis,:2]
            args = np.nonzero(connected)[0]
            if kind == 'intersection':
                keep = np.ones(len(chords), bool)
                for i,arg in enumerate(args):
                    in0 = __line_in_a_circle(pts[arg,:2], pt[:2], circles[:divider])
                    in1 = __line_in_a_circle(pts[arg,:2], pt[:2], circles[divider:])
                    if not (in0 and in1): keep[i] = False
                chords, args = chords[keep], args[keep]
            cen1 = circles[c1,:2]
            rad1 = pt[:2] - cen1
            cen0 = circles[c0,:2]
            rad0 = pt[:2] - cen0
            outvec = rad0/norm(rad0) + rad1/norm(rad1)
            ccw = __CCWmeasure(outvec,chords)
            arg = np.argmin(ccw)
            return args[arg]
        
    #traverse the intersection points to identify polygons
    polygons = []
    while len(xpts) > 0:
        #starting pt
        polygon = [xpts[0]]
        
        i = nextpt(xpts[0], xpts[1:]) + 1
        while True:
            pt = xpts[i]
            polygon.append(pt)
            xpts = np.delete(xpts, i, 0) 
            if all(polygon[-1] == polygon[0]): #back at the first point
                polygons.append(np.array(polygon))
                break
            i = nextpt(pt, xpts)
    
    #sum area of groups of circles (polygons and associated segments)
    areas = np.zeros(len(polygons))
    for i,polygon in enumerate(polygons):
        if len(polygon) == 3: #there must be a more elegant way to do this
            c0,c1 = polygon[0,2:]
            circs = circles[[c0,c1]]
            xpts = polygon[:2,:2]
            if kind == 'intersection':
                if sum([c0 < divider, c1 < divider]) != 1:
                    area = circle_circle_area(circs,kind='u',intersectpts=xpts)
                else:
                    area = circle_circle_area(circs,kind='x',intersectpts=xpts)
            else:
                area = circle_circle_area(circs,kind='u',intersectpts=xpts)
        else:
            area = __circpolyarea(circles, polygon, kind=kind, divider=divider)
        areas[i] = area
    
    return np.sum(areas)
    
def circle_intersection_pts(circles):
    """Idetifies all intersection points between the provided circles.
    
    circles is a numpy Nx3 array of x,y,r values for the circles, or the list
    equivalent.
    
    The function returns a list of the points and the index of the interseting
    circles an Nx4 array [[x,y,c0,c1],...]. It also returns a vector
    identifying whether each circle has any intersections and a nested list
    telling which circles each circle is completely within.
    """
    circles = np.array(circles)
    N = len(circles)
    
    #preallocate for N(N-1) = max possible number of intersections    
    xpts = np.zeros([N*(N-1),4])
    xflags = np.zeros(N, bool)
    incircs = [[] for i in range(N)]
    n = 0 #counter to index into xpts
    for i,circ0 in enumerate(circles):
        for j,circ1 in enumerate(circles[i+1:]):
            k = j+i+1
            pts = circle_circle_pts(*np.append(circ0[:3],circ1[:3]))
            if len(pts) == 1: #if one circle is inside another
                if pts == [0]: incircs[i].append(k)
                else:          incircs[k].append(i)
            if len(pts) > 1: #if the circles intersect
                xflags[[i,k]] = True
                pts = [list(pt)+[i,k] for pt in pts]
                xpts[n:n+2] = pts
                n += 2
    xpts = np.array(xpts[:n])
    
    return xpts, xflags, incircs

def __cArea(circles):
    return math.pi*circles[:,2]**2

def line_circle_pts(xc,yc,r,x0,y0,x1,y1):
    if x1 == x0:
        x = x0
        dx = abs(x - xc)
        if dx > r:
            return None
        elif dx == r:
            return x,yc
        else:
            A, B, C = 1.0, -2.0*yc, yc**2 + (x - xc)**2 - r**2
            radical = np.sqrt(B**2 - 4*A*C)
            yi0, yi1 = (-B + radical)/2.0/A, (-B - radical)/2.0/A
            return (x,yi0), (x,yi1)
    m = (y1 - y0)/(x1 - x0)
    b = y0 - x0*m
    y = lambda x: x*m + b
    A = m**2 + 1
    B = 2.0*(b - yc)*m - 2*xc
    C = xc**2 + (b - yc)**2 - r**2
    radicand = B**2 - 4*A*C
    if radicand < 0:
        return ()
    elif radicand == 0:
        xi = -B/2.0/A
        yi = y(xi)
        return (xi,yi),
    else:
        radical = np.sqrt(radicand)
        xi0, xi1 = (-B + radical)/2.0/A, (-B - radical)/2.0/A
        yi0, yi1 = map(y, [xi0, xi1])
        return (xi0, yi0), (xi1, yi1)

def __line_in_a_circle(pt0,pt1,circles):
    (x0,y0),(x1,y1) = pt0,pt1
    x,y,r = circles.T
    d0, d1 = dist(x0, y0, x, y), dist(x1, y1, x, y)
    D0, D1 = d0 - r, d1 - r
    incirc = np.logical_and(D0 < r/1e6, D1 < r/1e6)
    return any(incirc)
        
def __CCWmeasure(a,bs):
    """Gives a measure of how closely aligned the vectors in b are to a in a CCW sense. It
    does not compute the actual angle or even cos(angle) to save computation
    time. Instead, it returns a number that is greater the more closely aligned
    a b vectors are with a. The result is in the domain [-2,2], where greater
    numbers mean b is more CCW from a and 0 is at 180 deg.  bs is an Nx2 array.
    """
    ccw90 = np.array([-a[1], a[0]])
    bnorms2 = np.sum(bs**2, axis=1)
    anorm2 = np.sum(a**2)
    dots = np.sum(bs*a[np.newaxis,:], axis=1)
    cos_angle = dots/np.sqrt(bnorms2*anorm2)
    sin_sign = np.sum(bs*ccw90[np.newaxis,:], axis=1)
    past180 = (sin_sign < 0)
    before180 = np.logical_not(past180)
    cos_angle[before180] = cos_angle[before180] + 1.0
    cos_angle[past180] = -cos_angle[past180] - 1.0
    return -cos_angle

def norm(vec):
    return np.sqrt(np.sum(vec**2))