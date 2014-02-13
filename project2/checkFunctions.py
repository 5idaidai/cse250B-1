# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 13:03:32 2014

@author: sandicalhoun
"""

import numpy as np
from util import read_file
import ffs
import tags
from logistic_regression import LogisticRegression


"""Import a small sample dataset and run calcgis. Export the output to a csv."""

data_sample, labels_sample = read_file('sample')

lr = LogisticRegression(method="collins", max_iters=1)

labels_proc = lr.preproclabels(labels_sample)

i = int(np.random.rand() * len(data_sample))
n = len(data_sample[i])
ws = np.random.rand(ffs.numJ)
x = data_sample[i]
y = labels_proc[i]

#lr.calcgis(ws, x, n)
print data_sample[i]
print labels_sample[i],y
print ws

lr.calcAs(x, n)
print "As",lr.As


lr.calcBs()
print "Bs",lr.Bs

lr.calcS(ws, x, n)
print "S",lr.S


lr.calcgis(ws, x, n)
print lr.gis.shape
#print lr.gis
print lr.gis[0][2][len(tags.tags)-1]
print lr.gis[n-1][2][len(tags.tags)-1]

#np.savetxt("calcgisTest.csv", lr, delimiter=",", fmt='%1.4e')

#lr.calcalphas(ws, x, y, n)
#print "Alphas",lr.alphas

#lr.calcbetas(ws, x, y, n)
#print "Betas",lr.betas

lr.calcUMat(n)
print "U",lr.U

yhat = lr.calcYHat(x)
print yhat

expect = lr._calcCollExp(ws, x, n)
#print expect

print lr.calcYstar(y, n)
print y