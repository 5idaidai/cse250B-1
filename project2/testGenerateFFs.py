# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:34:23 2014

@author: axydes
"""

import os
import generateFFs as gFF
import tags

#question word at start of sentence yields question mark at end
def testffA1(x, i, n, a):
    return x[1] == a
    
def testffB1(yi1, yi, b):
    return yi == tags.tags[4]
    
#conjunction word at i, yields comma tag at i-1
def testffA2(x, i, n, a):
    return x[i] == a
    
def testffB2(yi1, yi, b):
    return yi1 == tags.tags[1]

#test case: last tag is a period
def testffA3(x, i, n, a):
    return 1

def testffB3(yi1, yi, b):
    return yi == tags.tags[tags.stop] and yi1 == tags.tags[2]
   
#default: tag is space
def testffA4(x, i, n, a):
    return i>0 and i<n-1

def testffB4(yi1, yi, b):
    return yi == tags.tags[6]

#Uppercase letter at beginning of word is associated with any tag
def testffA5(x, i, n, a):
    return i>0 and i<n-1 and x[i][0].isupper()

def testffB5(yi1, yi, b):
    return yi == b
    
#conjunction word at i, yields comma tag at i
def testffA6(x, i, n, a):
    return x[i] == a
    
def testffB6(yi1, yi, b):
    return yi == tags.tags[1]
    
#two commas in a row: should be negative weight as it's uncommon
def testffA7(x, i, n, a):
    return 1
    
def testffB7(yi1, yi, b):
    return yi1 == tags.tags[1] and yi == tags.tags[1]

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
    
    temp5 = {}
    temp5['aset'] = []
    temp5['bset'] = tags.tags
    temp5['A'] = testffA5
    temp5['B'] = testffB5
    ffs.append(temp5)
    
    temp6 = {}
    temp6['aset'] = tags.conj
    temp6['bset'] = []
    temp6['A'] = testffA6
    temp6['B'] = testffB6
    ffs.append(temp6)
    
    temp7 = {}
    temp7['aset'] = []
    temp7['bset'] = []
    temp7['A'] = testffA7
    temp7['B'] = testffB7
    ffs.append(temp7)
    
    gFF.generateFFs(ffs, os.path.splitext(os.path.basename(__file__))[0])
    #gFF.generateFFs(ffstest, os.path.splitext(os.path.basename(__file__))[0])
