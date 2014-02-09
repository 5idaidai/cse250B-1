from __future__ import division

import numpy as np
import ffs
import tags
from sklearn.utils.random import check_random_state
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

def log_sum_exp(x):
    m = x.max()
    x = x - m
    return m + np.log(np.exp(x).sum())
    
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
      
    @staticmethod      
    def log_sum_exp(x):
        m = x.max()
        newx=[]
        for xval in x:
            if xval != float("-inf"):
                newx.append(xval - m)
        return m + np.log(np.exp(newx).sum())
    
    @staticmethod
    def preproclabels(labels):
        newlabels = []
        
        i=0
        for y in labels:
            newy = []
            for tag in y:
                for ot in xrange(len(tags.tags)):
                    if tags.tags[ot] == tag:
                        newy.append(ot)
                        break
            newlabels.append(newy)
        
        return newlabels

    def fit(self, data, labels):
        self._validate_args()

        if self.method == "sgd" or self.method == "collins":
            betas = self._sgd(data, labels)
            
        self.coefficients_ = betas


    def calcYHat(self, x):
        end = len(x)-1
        yhat = ['']*len(x)
        pred = ['']*len(x)
        yhat[0] = tags.start
        pred[0] = tags.tags[tags.start]
#        yhat[end+1] = tags.tags[tags.stop]
        
        #last tag:
        yhat[end] = np.argmax(self.U[end])
        pred[end] = tags.tags[yhat[end]]
        
        for k in xrange(end-1,0,-1):
            temp = np.argmax((self.U[k][u] + self.gis[k+1][u][yhat[k+1]]) for u in xrange(self.m))
            yhat[k] = temp
            pred[k] = tags.tags[yhat[k]]
        
        return yhat
        

    def calcU(self, k, v):
        if k==1: #base case: return start tag
            return self.gis[k][tags.start][v]
        else:
            return max((self.U[k-1][u] + self.gis[k][u][v]) for u in xrange(self.m))


    def calcUMat(self, n):
        self.U = np.zeros((n, self.m))
        for k in xrange(1,n):
            for v in xrange(self.m):
                self.U[k][v] = self.calcU(k, v)
        return self.U


    def predict(self, data):
        self._validate_args()

        ld=len(data)
        predLabels = []
        for i in xrange(ld):
            x = data[i]
            n = len(x)
            self.calcgis(self.ws, x, n)
            self.calcUMat(n)
            predLabels.append(self.calcYHat(x))
        return predLabels
        
        
    def calcS(self, ws, x, n):
        tempS = []
        for j in xrange(ffs.numJ):
            if ffs.aFunc[j].func(x,1,n,ffs.aFunc[j].val) !=0:
                tempS.append(j)
        self.S = np.array(tempS)
        return self.S

    def sumFFs(self, ws, i, yi1, yi, x, n):
        summ = 0
        #um over all J feature functions
        for j in self.S:#range(0, ffs.numJ):
            summ += ws[j] * ffs.featureFunc[j](tags.tags[yi1], tags.tags[yi], x, i, n)
        return summ
        #return sum((ws[j] * ffs.featureFunc[j](tags.tags[yi1], tags.tags[yi], x, i, n)) for j in self.S)

    def calcgis(self, ws, x, n):
        #print "CalcGis start",datetime.now().time()
        #for i = 1 -> n (number of words)
        self.gis = np.zeros((n,self.m,self.m))
        for i in xrange(n):
            #for each pair of yi-1 yi
            for yi1 in xrange(self.m):
                for yi in xrange(self.m):                    
                    self.gis[i][yi1][yi] = self.sumFFs(ws, i, yi1, yi, x, n)
        #print "CalcGis stop",datetime.now().time()                    
        return self.gis
        

    def calcalpha(self, k, v):
        if k==0:
            return tags.tags[v] == tags.tags[tags.start]
        else:
            alph = self.alphas[k-1]
            gi = self.gis[k][:,v]
            temp = gi + alph
            return self.log_sum_exp(temp)

    def calcalphas(self, ws, x, y, n):
        for k in xrange(n):
            for v in xrange(self.m):
                self.alphas[k][v] = self.calcalpha(k,v)
        return
        

    def calcbeta(self, u, k, n):
        if k==n:#I(u==STOP)
            return tags.tags[u] == tags.tags[tags.stop]
        else:
            bet = self.betas[:,k-1]
            gi = self.gis[k+1][u]
            temp = gi + bet
            return self.log_sum_exp(temp)


    def calcbetas(self, ws, x, y, n):
        for k in xrange(n,0,-1):
            for u in xrange(self.m):
                self.betas[u][k] = self.calcbeta(u,k,n)
        return


    def calcZ(self, ws, x, y, n):
        zAlpha = sum(self.alphas[n][v] for v in xrange(self.m))
        zBeta = self.betas[tags.start][0]
        #print self.betas
        #print zAlpha, zBeta
        #assert zAlpha == zBeta    
        
        return zAlpha
        
        
    def calcF(self, j, x, y, n):
        #summ = 0
        #for i in range(1,n):
        #    summ = summ + ffs.featureFunc[j](y[i-1], y[i], x, i, n)
        return sum(ffs.featureFunc[j](y[i-1], y[i], x, i, n) for i in xrange(n))


    def _calcSGDExpect(self, ws, x, y, n):
        expect = np.zeros((len(ws)))
        for j in self.S:#range(0,len(ws)):
            summ = 0
            for i in xrange(n):
                for yi1 in xrange(self.m):
                    for yi in xrange(self.m):
                        num = self.alphas[i-1][yi1] + self.gis[i][yi1][yi] + self.betas[yi][i]
                        p = self.log_sum_exp(num) / self.Z
                        summ = summ + (ffs.featureFunc[j](tags.tags[yi1], tags.tags[yi], x, i, n) * p)
            expect[j] = summ
        return expect


    def _calcCollExp(self, ws, x, y, n):
        lws=len(ws)
        expect = np.zeros((lws))
        for j in xrange(lws):
            expect[j] = self.calcF(j, x, y, n)
        return expect


    def calcExpect(self, ws, x, y, n):
        if self.method == "sgd":
            return self._calcSGDExpect(ws, x, y, n)
        elif self.method == "collins":
            #calculate yhat
            self.calcUMat(n)
            yhat=self.calcYHat(x)
            return self._calcCollExp(ws, x, yhat, n)
        else:
            print "Incorrect method"
            return -1


    def _sgd_update(self, ws, x, y, rate):
        """single step in SGD"""
        n = len(x)
        
        #clear internal vars
        self.S = []
        self.gis = np.zeros((n,self.m,self.m))
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
        labels_valid=self.preproclabels(labels_valid)

        self.ws = np.zeros(ffs.numJ)
        rate = self.rate
        self.converged_ = False
        old_score = 0
        for epoch in xrange(self.max_iters):
            for i, (x, y) in enumerate(zip(data_train, labels_train)):
                self.ws = self._sgd_update(self.ws, x, y, rate)
            prediction = self.predict(data_valid)
            score = accuracy_score(labels_valid, prediction)
            print score,self.ws
            if score > 0 and score < old_score:#np.abs(score - old_score) < 1e-8:
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
        return self.ws

