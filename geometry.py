# -*- coding: utf-8 -*-
"""
Created on Sun Sep 07 13:39:08 2014

@author: Parke
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import pdb

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

def circle_area_union(*args, **kwargs):
    """Compute the area of the union of a set of circles. Can input as a list
    or Nx3 array of x,y,r values, or three separate lists or arrays. 
    """
    if len(args) == 1:
        x,y,r = np.array(args).T
    if len(args) == 3:
        x, y, r = map(np.array, args)
    brutegrid = int(kwargs['brutegrid']) if 'brutegrid' in kwargs.keys() else 0
    circles = np.array([x,y,r]).T
    N = len(circles)    
    
    if brutegrid:
        minx, maxx = np.min(x-r), np.max(x+r)
        miny, maxy = np.min(y-r), np.max(y+r)
        dx, dy = maxx - minx, maxy - miny
        area = dx*dy
        xvec = np.random.uniform(minx,maxx,brutegrid)
        yvec = np.random.uniform(miny,maxy,brutegrid)
        
        def incirc(x,y,circ):
            d = dist(x,y,circ[0],circ[1])
            return d <= circ[2]
        
        cnt = 0
        for x,y in zip(xvec,yvec):
            i = 0
            while True:
                if incirc(x,y,circles[i]):
                    cnt += 1
                    break
                else:
                    i += 1
                    if i > N-1: break
        return area*float(cnt)/brutegrid
    
    def out(pt,others):
        xi,yi = pt
        d = dist(xi,yi,x[others],y[others])
        return not any(d < r[others])
                    
    #find all outer intersection points for each circle
    #also use inorx to  record whether circle is in or intersects another
    xpts = np.zeros([N*(N-1)*2,4])
    inorx = np.zeros(N, bool)
    n = 0
    for i,circ0 in enumerate(circles):
        for j,circ1 in enumerate(circles[i+1:]):
            k = j+i+1
            pts = circle_intersect_pts(*np.append(circ0,circ1))
            if len(pts) == 1: #if one circle is inside another
                if pts == [0]: inorx[i] = True
                else:          inorx[k] = True
            if len(pts) > 1: #if the circles intersect
                inorx[[i,k]] = True
                #add points if they are outside of all other circles
                others = range(N)
                others.remove(i)
                others.remove(k)
                for pt in pts:
                    if out(pt, others):
                        xpts[n] = [pt[0],pt[1],i,k]
                        n += 1
    xpts = np.array(xpts[:n])
    
    #find nearest counterclockwise intersection pt on circcle c
    def nextpt(pt,c0,c1,pts,both=False):
        possible = np.logical_or(pts[:,2] == c1, pts[:,3] == c1)
        if both:
            possible2 = np.logical_or(pts[:,2] == c0, pts[:,3] == c0)
            possible = np.logical_or(possible, possible2)
        if sum(possible) == 0:
            raise ValueError('Well, that shouldn\'t have happend. Couldn\'t find a next point.')
        elif sum(possible) == 1:
            return np.nonzero(possible)[0][0]
        if sum(possible) > 1:
            cen0 = circles[c0,:2]
            cen1 = circles[c1,:2]
            radial0 = pt - cen0
            radial1 = pt - cen1
            outvec = radial0 + radial1
            chords = pts[possible,:2] - pt[np.newaxis,:]
            ccw = __CCWmeasure(outvec,chords)
            arg = np.argmax(ccw)
            return np.nonzero(possible)[0][arg]
    
    #traverse the intersection points to identify polygons
    polygons = []
    while len(xpts) > 0:
        #starting pt
        polygon = []
        polyapp = lambda i, c: polygon.append(list(xpts[i,:2]) + [c])
        pt = xpts[0,:2]
        clast, cnext = xpts[0,2:]
        # don't remove the starting point
        
        i = nextpt(pt, clast, cnext, xpts[1:], both=True) + 1
        clast = clast if clast in xpts[i,2:] else cnext
        polyapp(0, clast)
        while True:
            circs = xpts[i,2:]
            cnext = circs[circs != clast][0]
            pt = xpts[i,:2]
            polyapp(i, cnext)
            xpts = np.delete(xpts, i, 0) 
            if polygon[-1][:2] == polygon[0][:2]: #back at the first point
                polygons.append(polygon)
                break
            i = nextpt(pt, clast, cnext, xpts)
            clast = cnext
    
    #sum area of groups of circles (polygons and associate segments)
    areas = np.zeros(len(polygons))
    for i,polygon in enumerate(polygons):
        x,y,_ = zip(*polygon)
        area = polygon_area(x,y)
        line = len(polygon) == 3
        for j in range(len(polygon)-1):
            x0,y0,c = polygon[j]
            x1,y1,_ = polygon[j+1]
            xc,yc,r = circles[c]
            chord = np.array([x1-x0, y1-y0])
            nadir = np.array([[xc-x0, yc-y0]])
            segarea = circle_area_segment(r,x0,y0,x1,y1)
            left = (__CCWmeasure(chord,nadir)[0] > -1.0)
            area += segarea if (left and not line) else cArea(c) - segarea
        areas[i] = area
        
    #sum area of free circles
    free, = np.nonzero(np.logical_not(inorx))
    areas = np.append(areas, map(cArea, free))
        
    return sum(areas)
    

def circle_area_subtract(circles, subcircles):
    """Compute the area of the union of circles and subtract any area that
    overlaps with the circles in cubcircles.
    
    Circles are Nx3 arrays or the list equivalent. 
    
    Note that this is not the same as computing area of the difference of two
    shapes, since that would include area in the subtracted circles that does
    not overlap with the original circles.
    """

def __circle_group_areas(circles,xpts,add):
    """Compute the area of groups of intersecting circles by identifying the
    polygons constructed from xpts, computing the polygon area, and adding
    the area of the bordering circle segments.
    """
    

def circle_intersection_pts(circles, exclude=None):
    """Idetifies all intersection points between the provided circles.
    
    circles is a numpy NxM array where N is the number of circles (or list 
    equivalent). Each row must start with x,y,r, but can include additional
    values following to be used by a supplied exclude function
    
    A function can be provided in exclude that returns true when called with
    (pt, othercirlces), where othercircles are the circles that do not
    intersect at pt. When exclude returns True, the point will be excluded.
    
    The function returns a list of the points and the index of the interseting
    circles an Nx4 array [[x,y,c0,c1],...]. It also returns two vectors,
    the first identifying whether each circle has any intersections and the
    second whether that circle is completely inside of another circle
    """
    circles = np.array(circles)
    N = len(circles)
    
    #preallocate for N(N-1) = max possible number of intersections    
    xpts = np.zeros([N*(N-1),4])
    xflags, inflags = np.zeros(N, bool), np.zeros(N, bool)
    n = 0 #counter to index into xpts
    def apppt(pt,i,k,n):
        xpts[n] = [pt[0],pt[1],i,k]
        n += 1
    for i,circ0 in enumerate(circles):
        for j,circ1 in enumerate(circles[i+1:]):
            k = j+i+1
            pts = circle_circle_pts(*np.append(circ0[:3],circ1[:3]))
            if len(pts) == 1: #if one circle is inside another
                if pts == [0]: inflags[i] = True
                else:          inflags[k] = True
            if len(pts) > 1: #if the circles intersect
                xflags[[i,k]] = True
                #add points if they are outside of all other circles
                others = range(N)
                others.remove(i)
                others.remove(k)
                if exclude != None:
                    for pt in pts:
                        if not exclude(pt, circles(others)): apppt(pt,i,k,n)
                else: apppt(pt,i,k,n)
    xpts = np.array(xpts[:n])
    
    return xpts, xflags, inflags

def cArea(c):
    return math.pi*circles[c][2]**2

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
    