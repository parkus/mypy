# -*- coding: utf-8 -*-
"""
Created on Sun Sep 07 13:39:08 2014

@author: Parke
"""
import numpy as np
import math
import my_numpy as mnp

def dist(x0,y0,x1,y1):
    return np.sqrt((x1-x0)**2 + (y1-y0)**2)

def circle_area_segment(*args):
    """Compute the area of the segment of a circle given the radius and center-
    to-chord distance or the radius and the x0,y0,x1,y1 coordinates of the chord
    endpoints (useful for finding areas of circle-circle intersections when
    the intersection points are known)."""
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
    
def circle_area_union(circles, brutegrid=False):
    """Compute the area of the union of a set of circles, input as an Nx3 array
    or the list equivalent. Brutegrid uses a the random grid method (good for
    checkin and might be faster for large numbers of circles).
    """
    circles = np.array(circles)
    
    if brutegrid:
        def in_circles(x,y):
            for circ in circles:
                if dist(x,y,circ[0],circ[1]) <= circ[2]:
                    return True
            return False
        
        x,y,r = circles.T
        xrange = [np.min(x-r), np.max(x+r)]
        yrange = [np.min(y-r), np.max(y+r)]
        
        return brute_area(xrange, yrange, in_circles, brutegrid)
        
    def include(pt):
        xi,yi,c0,c1 = pt
        others = circles[[i for i in range(len(circles)) if i not in [c0,c1]]]
        d = dist(xi,yi,others[:,0],others[:,1])
        return not any(d < others[:,2])
                    
    xpts,xflags,incircs = circle_intersection_pts(circles,include)
    
    group_area = __circle_group_areas(circles,xpts)
    inflags = np.array([len(inc) > 0 for inc in incircs], bool)
    loners = np.logical_not(np.logical_or(inflags,xflags))
    loner_area = np.sum(__cArea(circles[loners]))
    
    return group_area+loner_area
    
def circleset_area_difference(circles0,circles1,xpts=None):
    circles = np.vstack([circles1,circles2])
    circles = np.hstack([circles, [[0]*len(circles0) + [1]*len(circles1)]])
    def include(pt):
        xi,yi,c0,c1 = pt
        if circles[c0][-1] == circles[c1][-1]:
             others = circles[[i for i in range(len(circles)) if i not in [c0,c1]]]
             d = dist(xi,yi,others[:,0],others[:,1])
             return not any(d < others[:,2])
    
                    
    xpts,xflags,incircs = circle_intersection_pts(circles,include)
    
    group_area = __circle_group_areas(circles,xpts)
    inflags = np.array([len(inc) > 0 for inc in incircs], bool)
    loners = np.logical_not(np.logical_or(inflags,xflags))
    loner_area = np.sum(__cArea(circles[loners]))
    
    return group_area+loner_area
    
def circleset_area_subtract(circles, subcircles, brutegrid=False):
    """Compute the area of the union of circles and subtract any area that
    overlaps with the circles in cubcircles.
    
    Circles are Nx3 arrays or the list equivalent. 
    
    Note that this is not the same as computing area of the difference of two
    shapes, since that would include area in the subtracted circles that does
    not overlap with the original circles.
    """
    Aunion = map(circle_area_union, [circles, subcircles])
    Adiff = circleset_area_difference(circles,subcircles)
    Aint = (sum(Aunion) - Adiff)/2.0
    return Aunion[0] - Aint

def circleset_area_intersect(circles1,circles2):
    Aunion = map(circle_area_union, [circles1, circles2])
    Adiff = circleset_area_difference(circles1,circles2)
    return (sum(Aunion) - Adiff)/2.0

def __circpolyarea(circles,polygon,outer='True'):
    n = len(polygon)    
    if n <= 3:
        raise ValueError('Polygon with 3 ot fewer vertices encountered.')
    x,y = polygon.T[:2]
    area = polygon_area(x,y)
    for j in range(n-1):
        x0,y0,c = polygon[j]
        x1,y1,_ = polygon[j+1]
        xc,yc,r = circles[c]
        segarea = circle_area_segment(r,x0,y0,x1,y1)
        chord = np.array([x1-x0, y1-y0])
        nadir = np.array([[xc-x0, yc-y0]])
        inside = (__CCWmeasure(chord,nadir)[0] > -1.0)
        area += segarea if inside else __cArea(circles[[c]]) - segarea
    
def __circle_group_areas(circles,xpts,kind='union',addsub=None):
    """Compute the area of groups of intersecting circles by identifying the
    polygons constructed from xpts, computing the polygon area, and adding
    the area of the bordering circle segments.
    
    The outer keyword specifies how to deal with single-line polygons (two points
    or three points when you include returning to the first). When true, the outer
    lenses are summed. When false, the inner lenses are summed.
    """
    def getconnector(c0,c1,c):
        if c0 in c:
            if c1 in c: return [c0,c1]
            return [c0]
        if c1 in c: return [c1]
        return []
    
    def poscon(connector):
        if len(connector) == 1:
            return not addsub[connector[0]]
        return False
        
    #find nearest counterclockwise intersection pt on circcle c
    def nextpt(pt,pts):
        c0,c1 = pt[2:]
        cs = pts[:,2:]
        connectors = [getconnector(c0,c1,c) for c in cs]
        connected = np.array([len(c) for c in connectors], bool)  
        if kind == 'subtraction':
            #make sure chords are within at least one positive and one
            #negative circle
            for i in np.nonzero(connected)[0]:
                if len(connectors[i]) == 2:
                    connected[i] = bool(sum(addsub[connectors[i]]))
                else:
                    c = connectors[i][0]
                    othercircs = circles[addsub != addsub[c]]
                    connected[i] = __line_in_a_circle(pts[i,:2],pt[:2],
                                                      othercircs)          
        if sum(connected) == 0:
            raise ValueError('Well, that shouldn\'t have happend. Couldn\'t find a next point.')
        elif sum(connected) == 1:
            return np.nonzero(connected)[0][0]
        if sum(connected) > 1:
            chords = pts[connected,:2] - pt[np.newaxis,:2]
            cen1 = circles[c1,:2]
            rad1 = pt[:2] - cen1
            cen0 = circles[c0,:2]
            rad0 = pt[:2] - cen0
            outvec = rad0/norm(rad0) + rad1/norm(rad1)
            ccw = __CCWmeasure(outvec,chords)
            arg = np.argmax(ccw)
            return np.nonzero(connected)[0][arg]
        
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
            x0,y0,c0,c1 = polygon[0]
            x1,y1,_,_ = polygon[1]
            r0, r1 = [circles[c,2] for c in [c0,c1]]
            mid = np.array([(x0+x1)/2.0, (y0+y1)/2.0])
            d0, d1 = [mid - circles[c,:2] for c in [c0,c1]]
            opposite = any(d0/d1 < 0)
            segs = [circle_area_segment(r,x0,y0,x1,y1) for r in [r0,r1]]
            circs = __cArea(circles[[c0,c1]])
            if kind == 'union':
                if opposite:
                    area = sum(circs) - sum(segs)
                else:
                    area = max(circs) - min(segs) + max(segs)
            if kind == 'subtraction':
                if opposite:
                    area = sum(segs)
                else:
                    area = min(circs) - max(segs) + min(segs)
        else:
            x,y = polygon.T[:2]
            area = polygon_area(x,y)
            n = len(polygon)
            for j in range(n-1):
                x0,y0,c00,c01 = polygon[j]
                x1,y1,c10,c11 = polygon[j+1]
                if c00 in [c10,c11] and c01 in [c10,c11]:
                    c2s = polygon[((j+2) % (n-1)),2:]
                    c = c00 if c00 not in c2s else c01
                else :
                    c = c00 if c00 in [c10,c11] else c01
                xc,yc,r = circles[c]
                segarea = circle_area_segment(r,x0,y0,x1,y1)
                chord = np.array([x1-x0, y1-y0])
                nadir = np.array([[xc-x0, yc-y0]])
                inside = (__CCWmeasure(chord,nadir)[0] > -1.0)
                area += segarea if inside else __cArea(circles[[c]]) - segarea
        areas[i] = area
    
    return np.sum(areas)
    
def circle_intersection_pts(circles, include='all'):
    """Idetifies all intersection points between the provided circles.
    
    circles is a numpy NxM array where N is the number of circles (or list 
    equivalent). Each row must start with x,y,r, but can include additional
    values following to be used by a supplied exclude function
    
    A function can be provided in exclude that returns true when called with
    (pt, othercirlces), where othercircles are the circles that do not
    intersect at pt. When exclude returns True, the point will be excluded.
    
    The function returns a list of the points and the index of the interseting
    circles an Nx4 array [[x,y,c0,c1],...]. It also returns a vector
    identifying whether each circle has any intersections and a nested list
    telling which circles each circle is completely within.
    """
    if include == 'all': include = lambda: True
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
                #add points if they meet the include function criterion
                pts = [list(pt)+[i,k] for pt in pts]
                for pt in pts:
                    if include(pt): 
                        xpts[n] = pt
                        n += 1
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
    

def circle_circle_pts(x0,y0,r0,x1,y1,r1):
    """Computes the points where two circles interesect.
    
    If the circles do not interset, an empty list is returned. If one circle is
    enclosed in another, the index ([0] or [1]) of that circle is returned.
    Otherwise, the intersection points are returned as the tuple (xi0,yi0),
    (xi1,yi1) in no particular order.
    """
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
        return (xi0,yi0), (xi1,yi1)

def __line_in_a_circle(pt0,pt1,circles):
    for xc,yc,r in circles:
        (x0,y0),(x1,y1) = pt0,pt1
        ipts = line_circle_pts(xc,yc,r,x0,y0,x1,y1)
        if len(ipts) == 2:
            (xi0,yi0),(xi1,yi1) = ipts
            xs_in = all(mnp.inranges([x0,x1],[xi0,xi1]))
            ys_in = all(mnp.inranges([y0,y1],[yi0,yi1]))
            if xs_in and ys_in:
                return True
    return False
        
def __CCWmeasure(a,bs):
    """Gives a measure of how closely aligned the vectors in b are to a in a CCW sense. It
    does not compute the actual angle or even cos(angle) to save computation
    time. Instead, it returns a number that is greater the more closely aligned
    a b vectors are with a. If they are more than 180 deg CCW from a, the 
    number is negative. bs is an Nx2 array."""
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
    return cos_angle

def norm(vec):
    return np.sqrt(np.sum(vec**2))