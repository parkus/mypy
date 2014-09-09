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

def circle_intersect_pts(x0,y0,r0,x1,y1,r1):
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
    plot = kwargs['plot'] if 'plot' in kwargs.keys() else False
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
        
    cArea = lambda c: math.pi*circles[c][2]**2
    
    if plot:
        fig = plt.figure()
        __plotcircs(circles)
    
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
    def nextpt(pt,c0,c1,pts):
        x,y = pt
        xc,yc,r = circles[c]
        possible = np.logical_or(pts[:,2] == c, pts[:,3] == c)
        
        if sum(possible) == 0:
            raise ValueError('Well, that shouldn\'t have happend. Couldn\'t find a next point.')
        elif sum(possible) == 1:
            return np.nonzero(possible)[0][0]
        if sum(possible) > 1:
            nadir = [xc - x, yc - y]
            tangent = np.array([nadir[1], -nadir[0]])
            chords = pts[possible,:2] - np.array([x,y])[np.newaxis,:]
            ccw = __CCWmeasure(tangent,chords)
            arg = np.argmax(ccw)
            return np.nonzero(possible)[0][arg]
    
    #traverse the intersection points to identify polygons
    polygons = []
    while len(xpts) > 0:
        #starting pt
        polygon = []
        polyapp = lambda i, c: polygon.append(list(xpts[i,:2]) + [c])
        pt = xpts[0,:2]
        c0,c1 = xpts[0,2:]
        # don't remove the starting point yet
        
        #get nextpt for both c0 and c1
        i0 = nextpt(pt,c0,xpts[1:]) + 1
        i1 = nextpt(pt,c1,xpts[1:]) + 1
        if i0 == i1:
        #then just the two circles intersect and the polygon is a line
            polyapp(0,c0)
            polyapp(i0,c1)
            xpts = np.delete(xpts, [0,i0], 0)
            break
        #otherwise pick whichever does not lie on the same two circles
        first = all(xpts[i1,2:] == [c0,c1]) or all(xpts[i1,2:] == [c1,c0])
        i = i0 if first else i1
        clast = c0 if first else c1
        cnext = xpts[i,2] if xpts[i,2] != clast else xpts[i,3]
        polyapp(0,clast)
        polyapp(i,cnext)
        pt = xpts[i,:2]
        xpts = np.delete(xpts, i, 0)
        
        #now get the rest
        while True:
            i = nextpt(pt, cnext, xpts)
            clast = cnext
            cnext = xpts[i,2] if xpts[i,2] != clast else xpts[i,3]
            pt = xpts[i,:2]
            polyapp(i, cnext)
            xpts = np.delete(xpts, i, 0)
            if polygon[-1][:2] == polygon[0][:2]: #back at the first point
                polygons.append(polygon)
                break
    
    #sum area of groups of circles (polygons and associate segments)
    areas = []
    for polygon in polygons:
        if len(polygon) == 2:
            c0,c1 = polygon[0][2:]
            r0,r1 = circles[c0[2]], circles[c1[2]]
            x0,y0 = polygon[0][:2]
            x1,y1 = polygon[1][:2]
            area = cArea(c0) + cArea(c1)
            area -= circle_area_segment(r0,x0,y0,x1,y1)
            area -= circle_area_segment(r1,x0,y0,x1,y1)
        else:
            x,y,_ = zip(*polygon)
            area = polygon_area(x,y)
            for i in range(len(polygon)-1):
                x0,y0,c = polygon[i]
                x1,y1,_ = polygon[i+1]
                xc,yc,r = circles[c]
                chord = [x1-x0, y1-y0]
                perp = np.array([-chord[1],chord[0]])
                nadir = np.array([[xc-x0, yc-y0]])
                segarea = circle_area_segment(r,x0,y0,x1,y1)
                left = (__CCWmeasure(perp,nadir)[0] > 0)
                if left: area += segarea
                else: area += cArea(c) - segarea
        areas.append(area)
        
    #sum area of free circles
    free, = np.nonzero(np.logical_not(inorx))
    areas.extend(map(cArea, free))
        
    return sum(areas)
    
def __plotcircs(circles):
    for i,circ in enumerate(circles):
        theta = np.linspace(0,2*math.pi,100)
        x,y,r = circ
        dx,dy = r*np.cos(theta), r*np.sin(theta)
        xv,yv = x+dx, y+dy
        plt.plot(xv,yv)
        plt.text(x,y,i,ha='center',va='center')

def circle_area_arbitrary():
    pass

def __CCWmeasure(a,bs):
    """Gives a measure of how closely aligned the vectors in b are to a in a CCW sense. It
    does not compute the actual angle or even cos(angle) to save computation
    time. Instead, it returns a number that is greater the more closely aligned
    a b vectors are with a. If they are more than 180 deg CCW from a, the 
    number is negative. bs is an Nx2 array."""
    norms2 = np.sum(bs**2, axis=1)
    dots = np.sum(bs*a[np.newaxis,:], axis=1)
    return dots/norms2