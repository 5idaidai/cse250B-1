from __future__ import division

import numpy as np
import ffs
from scipy.optimize import fmin_l_bfgs_b
from sklearn.utils.random import check_random_state

def log_sum_exp(x):
    m = x.max()
    x = x - m
    return m + np.log(np.exp(x).sum())


def log_prob(y, xb):
    """returns log prob(Y=y)

    xb = np.dot(X, betas)

    """
    if y == 1:
        xb = -xb
    return -log_sum_exp(np.array([0, xb]))


def lcl(data, labels, betas):
    """log conditional likelihood"""
    return sum(log_prob(y, np.dot(x, betas)) for x, y in zip(data, labels))


def rlcl(data, labels, betas, mu):
    """regularized log conditional likelihood"""
    return lcl(data, labels, betas) - mu * sum(np.power(betas[1:], 2))


def lcl_prime(data, labels, betas):
    """gradient of lcl"""
    coeffs = np.array(list(y - np.exp(log_prob(1, np.dot(x, betas)))
                           for x, y in zip(data, labels)))
    return coeffs.dot(data)


def rlcl_prime(data, labels, betas, mu):
    """gradient of rlcl"""
    grad = lcl_prime(data, labels, betas)
    grad[1:] = grad[1:] - 2 * mu * betas[1:]
    return grad


class LogisticRegression(object):
    """Logistic regression model with L2 regularization..

    Parameters
    ----------
    method: string
        May be "sgd" or "lbfgs".

    mu: float
        Strength of regularization.

    rate: float
        Learning rate of SGD.

    decay: float
        After every epoch, ``rate`` is reduced to ``decay * rate``.

    max_iters: int
        Maximum iterations of SGD.

    random_state: None or int or numpy.random.RandomState
        Seed for randomness

    """

    def __init__(self, method="sgd", mu=0.1, rate=1, decay=1,
                 max_iters=1000, random_state=None):
        self.method = method
        self.mu = mu
        self.rate = rate
        self.decay = decay
        self.max_iters = 1000
        self.random_state = random_state
        self.gis = []
        self.alphas = []
        self.betas = []
        self.Z = []

    def _validate_args(self):
        methods = ("sgd", "collins", "lbfgs")
        if self.method not in methods:
            raise Exception("method '{}' invalid. should be"
                            " one of {}".format(self.method, methods))
        if self.mu <= 0:
            raise Exception("invalid regularization strength. mu={},"
                            " but it should be > 0".format(self.mu))

        if self.rate <= 0:
            raise Exception("invalid step schedule. rate={},"
                            " but it should be > 0".format(self.rate))

        if self.decay <= 0 or self.decay > 1:
            raise Exception("invalid decay: {}."
                            " should be in (0, 1]".format(self.decay))

        if self.max_iters <= 0:
            raise Exception("invalid max_iters: {}".format(self.max_iters))

    def fit(self, X, labels):
        self._validate_args()

        if self.method == "sgd" or self.method == "collins":
            betas = self._sgd(X, labels)
        else:
            betas = self._lbfgs(X, labels)

        self.intercept_ = betas[0]
        self.coefficients_ = betas[1:]

    def predict(self, X):
        self._validate_args()
        XB = self.intercept_ + np.dot(X, self.coefficients_)
        XB = XB.reshape(-1, 1)
        f = lambda z: log_prob(1, z)
        probs = np.exp(np.apply_along_axis(f, axis=1, arr=XB))
        neg, pos = self.old_labels_
        return np.where(probs.ravel() >= 0.5, pos, neg)


    def calcgis(self, ws, x, y):
        #for i = 1 -> n (number of words)
            #for each pair of yi-1 yi
        return []


    def calcalphas(self, ws, x, y):
        return []


    def calcbetas(self, ws, x, y):
        return []


    def calcZ(self, ws, x, y):
        return []


    def _calcExpect(self, ws, x, y):
        return 0


    def _calcCollExp(self, ws, x, y):
        return 0


    def calcExpect(self, ws, x, y):
        if self.method == "sgd":
            return self._calcExpect(ws, x, y)
        elif self.method == "collins":
            return self._calcCollExp(ws, x, y)
        else:
            print "Incorrect method"
            return -1


    def _sgd_update(self, ws, x, y, rate):
        """single step in SGD"""
        
        #clear internal vars
        self.gis = []
        self.alphas = []
        self.betas = []
        self.Z = []
        
        #calculate gi matrices
        self.gis = self.calcgis(ws, x, y)

        #calculate forward(alpha) & backward(beta) vectors, and Z
        self.alphas = self.calcalphas(ws, x, y)
        self.betas = self.calcbetas(ws, x, y)
        self.Z = self.calcZ(ws, x, y)

        #compute expectation
        expectation = self.calcExpect(ws, x, y)        
        
        p = np.exp(log_prob(1, np.dot(x, ws)))
        result = ws + rate * ((y - p) * x - 2 * self.mu * ws)
        # do not regularize intercept
        result[0] = ws[0] + rate * ((y - p) * x[0])
        return result


    def _sgd(self, data, labels):
        # shuffle data
        n = len(data)
        idx = np.arange(n)
        state = check_random_state(self.random_state)
        state.shuffle(idx)
        data = data[idx]
        labels = labels[idx]

        ws = np.zeros(ffs.numJ)
        rate = self.rate
        self.converged_ = False
        for epoch in range(self.max_iters):
            old_lcl = rlcl(data, labels, ws, self.mu)
            for i, (x, y) in enumerate(zip(data, labels)):
                betas = self._sgd_update(ws, x, y, rate)
            new_lcl = rlcl(data, labels, ws, self.mu)
            if np.abs(new_lcl - old_lcl) < 1e-8:
                self.converged_ = True
                break
            rate = rate * self.decay
        if self.converged_:
            print "converged after {} epochs".format(epoch)
        else:
            print "did not converge"

        self.lcl_ = lcl(data, labels, betas)
        self.rlcl_ = rlcl(data, labels, betas, self.mu)
        return betas


    def _lbfgs(self, data, labels):
        f = lambda b: -rlcl(data, labels, b, self.mu)
        fprime = lambda b: -rlcl_prime(data, labels, b, self.mu)
        x0 = np.zeros(data.shape[1])
        result = fmin_l_bfgs_b(f, x0, fprime)
        print result[2]['warnflag'], result[2]['task']
        betas = result[0]

        self.lcl_ = lcl(data, labels, betas)
        self.rlcl_ = rlcl(data, labels, betas, self.mu)

        return betas
