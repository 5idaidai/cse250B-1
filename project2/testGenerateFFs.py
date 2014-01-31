# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:34:23 2014

@author: axydes
"""

import generateFFs as gFF

tags = []
tags.append("START")
tags.append("STOP")
tags.append("COMMA")
tags.append("PERIOD")
tags.append("QUESTION_MARK")
tags.append("EXCLAMATION_POINT")
tags.append("COLON")
tags.append("SPACE")

def testffA1(x, i, n, a):
    return x[0] == 'W'
    
def testffB1(yi1, yi, b):
    return yi == tags[4]
    
def testffA2(x, i, n, a):
    return x[i][0].isupper()
    
def testffB2(yi1, yi, b):
    return yi == b


if __name__ == "__main__":
    ffs = []
        
    temp1 = {}
    temp1['aset'] = []
    temp1['bset'] = []
    temp1['A'] = testffA1
    temp1['B'] = testffB1
    ffs.append(temp1)
    
    temp2 = {}
    temp2['aset'] = []
    temp2['bset'] = tags
    temp2['A'] = testffA2
    temp2['B'] = testffB2
    ffs.append(temp2)
    
    gFF.generateFFs(ffs, os.path.splitext(os.path.basename(__file__))[0])
