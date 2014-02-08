from __future__ import division

import numpy as np
import ffs
import tags
from sklearn.utils.random import check_random_state
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

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
        self.max_iters = max_iters
        self.random_state = random_state
        self.m = len(tags.tags)
        self.gis = []
        self.alphas = np.zeros((self.m,self.m))
        self.betas = np.zeros((self.m,self.m))
        self.Z = 1


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

    def fit(self, data, labels):
        self._validate_args()

        if self.method == "sgd" or self.method == "collins":
            betas = self._sgd(data, labels)
            
        self.coefficients_ = betas


    def calcYHat(self, x):
        end = len(x)-2
        yhat = ['']*len(x)
        yhat[0] = tags.tags[0]
        yhat[end+1] = tags.tags[1]
        
        #last tag:
        yhat[end] = tags.tags[np.argmax(self.U[end])]
        
        for u in range(end,0,-1):
            yhat[u] = tags.tags[np.argmax(self.U[u])]
        
        return yhat
        

    def calcU(self, k, v):
        if k==0: #base case: return start tag
            return self.gis[k][0][v]
        else:
            return max((self.U[k-1][yk1] + self.gis[k][yk1][v]) for yk1 in range(self.m))


    def calcUMat(self, n):
        self.U = np.zeros((n, self.m))
        for k in range(n):
            for v in range(self.m):
                self.U[k][v] = self.calcU(k, v)
        return self.U


    def predict(self, data):
        self._validate_args()

        predLabels = []        
        for i in range(len(data)):
            x = data[i]
            n = len(x)
            self.calcgis(self.ws, x, n)
            self.calcUMat(n)
            predLabels.append(self.calcYHat(x))
        return predLabels
        
        
    def calcS(self, ws, x, n):
        for j in range(ffs.numJ):
            if ffs.aFunc[j].func(x,1,n,ffs.aFunc[j].val) !=0:
                self.S.append(j)
        return self.S


    def calcgis(self, ws, x, n):
        #for i = 1 -> n (number of words)
        for i in range(0, n):
            #compute gi
            self.gis.append(np.zeros((self.m,self.m)))
            #for each pair of yi-1 yi
            for yi1 in range(0, self.m):
                for yi in range(0, self.m):
                    summ = 0
                    #sum over all J feature functions
                    for j in self.S:#range(0, ffs.numJ):
                        summ += ws[j] * ffs.featureFunc[j](tags.tags[yi1], tags.tags[yi], x, i, n)
                    self.gis[i][yi1][yi] = summ
        return self.gis
        

    def calcalpha(self, k, v):
        if k==0:
            return tags.tags[v] == tags.tags[0]
        else:
            #summ = 0
            #for u in range(0,self.m):
            #    summ = summ + (self.alphas[k-1][u] * np.exp(self.gis[k][u][v]))
            #return summ
            return sum((self.alphas[k-1][u] * np.exp(self.gis[k][u][v])) for u in range(0,self.m))


    def calcalphas(self, ws, x, y, n):
        for k in range(0, n):
            for v in range(0, self.m):
                self.alphas[k][v] = self.calcalpha(k,v)
        return
        

    def calcbeta(self, u, k, n):
        if k==n:#I(u==STOP)
            return tags.tags[u] == tags.tags[1]
        else:
            #summ = 0
            #for v in range(0,self.m):
            #    summ = summ + (np.exp(self.gis[k+1][u][v]) * self.betas[v][k+1])
            #return summ
            return sum((np.exp(self.gis[k+1][u][v]) * self.betas[v][k+1]) for v in range(0,self.m))


    def calcbetas(self, ws, x, y, n):
        for k in range(n,0,-1):
            for u in range(0, self.m):
                self.betas[u][k] = self.calcbeta(u,k,n)
        return


    def calcZ(self, ws, x, y, n):
        zAlpha = sum(self.alphas[n][v] for v in range(0,self.m))
        zBeta = self.betas[0][0]
        #print self.betas
        #print zAlpha, zBeta
        #assert zAlpha == zBeta    
        
        return zAlpha
        
        
    def calcF(self, j, x, y, n):
        #summ = 0
        #for i in range(1,n):
        #    summ = summ + ffs.featureFunc[j](y[i-1], y[i], x, i, n)
        return sum(ffs.featureFunc[j](y[i-1], y[i], x, i, n) for i in range(0,n))


    def _calcSGDExpect(self, ws, x, y, n):
        expect = np.zeros((len(ws)))
        for j in self.S:#range(0,len(ws)):
            summ = 0
            for i in range(0,n):
                for yi1 in range(0, self.m):
                    for yi in range(0, self.m):
                        num = self.alphas[i-1][yi1] * np.exp(self.gis[i][yi1][yi]) * self.betas[yi][i]
                        p = num / self.Z
                        summ = summ + (ffs.featureFunc[j](tags.tags[yi1], tags.tags[yi], x, i, n) * p)
            expect[j] = summ
        return expect


    def _calcCollExp(self, ws, x, y, n):
        expect = np.zeros((len(ws)))
        for j in range(0, len(ws)):
            expect[j] = self.calcF(j, x, y, n)
        return expect


    def calcExpect(self, ws, x, y, n):
        if self.method == "sgd":
            return self._calcSGDExpect(ws, x, y, n)
        elif self.method == "collins":
            return self._calcCollExp(ws, x, y, n)
        else:
            print "Incorrect method"
            return -1


    def _sgd_update(self, ws, x, y, rate):
        """single step in SGD"""
        n = len(x)
        
        #clear internal vars
        self.S = []
        self.gis = []
        self.alphas = np.zeros((n,self.m))
        self.betas = np.zeros((self.m,n))
        self.Z = 1
        
        #calculate S set (set of feature functions that aren't 0)
        self.calcS(ws, x, n)
        #print len(self.S),self.S
        
        #calculate gi matrices
        self.calcgis(ws, x, n)

        #calculate forward(alpha) & backward(beta) vectors, and Z
        self.calcalphas(ws, x, y, n)
        self.calcbetas(ws, x, y, n-1)
        self.Z = self.calcZ(ws, x, y, n-1)

        #compute expectation
        fval = self._calcCollExp(ws, x, y, n)
        expectation = self.calcExpect(ws, x, y, n)        
        
        #p = np.exp(log_prob(1, np.dot(x, ws)))
        result = ws + rate * (fval - expectation)#- 2 * self.mu * ws)
        # do not regularize intercept
        #result[0] = ws[0] + rate * ((y - p) * x[0])
        return result


    def _sgd(self, data, labels):
        #split off validation set
        data_train, data_valid, labels_train, labels_valid = \
            train_test_split(data, labels, test_size=0.3, random_state=0)        
        
        # shuffle data
        n = len(data_train)
        idx = np.arange(n)
        state = check_random_state(self.random_state)
        state.shuffle(idx)
        data_train = data_train[idx]
        labels_train = labels_train[idx]

        ws = np.zeros(ffs.numJ)
        rate = self.rate
        self.converged_ = False
        old_score = 0
        for epoch in range(self.max_iters):
            for i, (x, y) in enumerate(zip(data_train, labels_train)):
                ws = self._sgd_update(ws, x, y, rate)
            self.ws = ws
            prediction = self.predict(data_valid)
            score = accuracy_score(labels_valid, prediction)
            if score > 0 and np.abs(score - old_score) < 1e-8:
                self.converged_ = True
                break
            rate = rate * self.decay
            old_score = score
        if self.converged_:
            print "converged after {} epochs".format(epoch)
        else:
            print "did not converge"

        #self.lcl_ = lcl(data, labels, betas)
        #self.rlcl_ = rlcl(data, labels, betas, self.mu)
        return ws

