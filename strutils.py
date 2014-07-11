# -*- coding: utf-8 -*-
"""
Created on Mon Jun 02 23:04:47 2014

@author: Parke
"""

def pretty_deltatime(seconds):
    if seconds < 120.0:
        return '{:.1f} s'.format(seconds)
    minutes = seconds/60.0
    if minutes < 60.0:
        return '{:.1f} min'.format(minutes)
    hours = minutes/60.0
    if hours < 24.0:
        return '{:.1f} h'.format(hours)
    days = hours/24.0
    if days < 365.0:
        return '{:.1f} d'.format(days)
    years = days/365.0
    return '{:.1f} y'.format(years)
    
def parse_folder(pathstr):
    if pathstr.count('\\') > 0:
        pieces = pathstr.split('\\')
        return '\\'.join(pieces[0:-1]) + '\\'
    elif pathstr.count('/') > 0:
        pieces = pathstr.split('/')
        return '/'.join(pieces[0:-1]) + '/'
    else:
        return './'
    