# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 13:03:32 2014

@author: sandicalhoun
"""

import numpy as np
from util import read_file
from logistic_regression import LogisticRegression


"""Import a small sample dataset and run calcgis. Export the output to a csv."""

data_sample, labels_sample = read_file('sample')

n = len(data_sample)
ws = np.random.rand(n)
x = data_sample

checkgis = LogisticRegression(method="collins", max_iters=1)

#checkgis.calcgis(ws, x, n)
print data_sample
print ws
print checkgis.calcgis(ws, x, n)
print checkgis.gis.shape

#np.savetxt("calcgisTest.csv", checkgis, delimiter=",", fmt='%1.4e')





