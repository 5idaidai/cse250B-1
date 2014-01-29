# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:34:23 2014

@author: axydes
"""

import generateFFs as gFF

def testffA1(x, i, n, a):
    return x[0] == 'W'
    
def testffB1(yi1, yi, b):
    return yi == '?'

if __name__ == "__main__":
    ffs = []
    
    temp1 = {}
    temp1['aset'] = []
    temp1['bset'] = [2,3]
    temp1['A'] = testffA1
    temp1['B'] = testffB1
    ffs.append(temp1)
    
    temp2 = {}
    temp2['aset'] = [3,4,5]
    temp2['bset'] = ['a','b','c']
    temp2['A'] = testffA1
    temp2['B'] = testffB1
    ffs.append(temp2)
    
    gFF.generateFFs(ffs, os.path.splitext(os.path.basename(__file__))[0])
