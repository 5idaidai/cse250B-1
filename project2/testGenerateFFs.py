# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:34:23 2014

@author: axydes
"""

import os
import generateFFs as gFF
import tags

def testffA1(x, i, n, a):
    return x[1] == a
    
def testffB1(yi1, yi, b):
    return yi == tags.tags[4]
    
def testffA2(x, i, n, a):
    return x[i] == a
    
def testffB2(yi1, yi, b):
    return yi1 == tags.tags[1]

def testffA3(x, i, n, a):
    return 1

#test case: last tag is a period
def testffB3(yi1, yi, b):
    return yi == tags.tags[tags.stop] and yi1 == tags.tags[2]
    
def testffA4(x, i, n, a):
    return i<n

def testffB4(yi1, yi, b):
    return yi == tags.tags[6]



if __name__ == "__main__":
    ffs = []
    ffstest = []
        
    temp1 = {}
    temp1['aset'] = tags.quest
    temp1['bset'] = []
    temp1['A'] = testffA1
    temp1['B'] = testffB1
    ffs.append(temp1)
    
    temp2 = {}
    temp2['aset'] = tags.conj
    temp2['bset'] = []
    temp2['A'] = testffA2
    temp2['B'] = testffB2
    ffs.append(temp2)
    
    temp3 = {}
    temp3['aset'] = []
    temp3['bset'] = []
    temp3['A'] = testffA3
    temp3['B'] = testffB3
    ffs.append(temp3)
    ffstest.append(temp3)
    
    temp4 = {}
    temp4['aset'] = []
    temp4['bset'] = []
    temp4['A'] = testffA4
    temp4['B'] = testffB4
    ffs.append(temp4)
    ffstest.append(temp4)
    
    gFF.generateFFs(ffs, os.path.splitext(os.path.basename(__file__))[0])
    #gFF.generateFFs(ffstest, os.path.splitext(os.path.basename(__file__))[0])
