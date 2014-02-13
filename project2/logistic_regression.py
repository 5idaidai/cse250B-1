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
        self.As = np.zeros((ffs.numJ,10))
        self.Bs = np.zeros((ffs.numJ,self.m,self.m))
        self.gis = []
        self.alphas = np.zeros((self.m,self.m))
        self.betas = np.zeros((self.m,self.m))
        self.Z = 1


    def _validate_args(self):
        methods = ("sgd", "collins", "cd")
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
        
        for y in labels:
            newy = []
            for tag in y:
                newy.append(tags.tags.index(tag))
            newlabels.append(newy)
        
        return newlabels

    def fit(self, data, labels):
        self._validate_args()

        betas = self._sgd(data, labels)
            
        self.coefficients_ = betas


    def calcYHat(self, x):
        end = len(x)-1
        yhat = ['']*len(x)
        pred = ['']*len(x)
        yhat[0] = tags.start
        pred[0] = tags.tags[tags.start]
        #yhat[end+1] = tags.stop  
        #pred[end+1] = tags.tags[tags.stop]
        yhat[end] = tags.stop  
        pred[end] = tags.tags[tags.stop]
        
        #last tag:
        yhat[end-1] = np.argmax(self.U[end-1])
        pred[end-1] = tags.tags[yhat[end-1]]
        
        for k in xrange(end-2,0,-1):
            temp = np.argmax(self.U[k] + self.gis[k+1][:,yhat[k+1]])
            yhat[k] = temp
            pred[k] = tags.tags[yhat[k]]
            

        
        return yhat
        

    def calcU(self, k, v):
        if k==1: #base case: return start tag
            return self.gis[k][tags.start][v]
        else:
            return max((self.U[k-1] + self.gis[k][:,v]))


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
            self.calcAs(x, n)
            self.calcBs()
            self.calcgis(self.ws, x, n)
            self.calcUMat(n)
            predLabels.append(self.calcYHat(x))
        return predLabels
        
    
    def calcAs(self, x, n):
        self.As = np.fromiter((ffs.aFunc[j].func(x,i,n, ffs.aFunc[j].val)
                            for j in xrange(ffs.numJ)
                            for i in xrange(n)),
                            dtype=np.float,
                            count=ffs.numJ*n)
        self.As = self.As.reshape((ffs.numJ,n))
        return self.As

        
    
    def calcBs(self):
        self.Bs = np.fromiter((ffs.bFunc[j].func(tags.tags[yi1], tags.tags[yi], ffs.bFunc[j].val)
                    for j in xrange(ffs.numJ)
                    for yi1 in xrange(self.m)
                    for yi in xrange(self.m)),
                    dtype=np.float,
                    count=ffs.numJ*self.m*self.m)
        self.Bs = self.Bs.reshape((ffs.numJ,self.m,self.m))
        return self.Bs                   
        
    
    def calcS(self, ws, x, n):
        self.S = []
        tempS = []
        for j in xrange(ffs.numJ):
            if max(self.As[j]) !=0:
                tempS.append(j)
        self.S = np.array(tempS)
        return self.S
    
    def calcgis(self, ws, x, n):
        self.gis = np.fromiter((sum(ws * self.As[:,i] * self.Bs[:,yi1,yi])
                            for i in xrange(n) 
                            for yi1 in xrange(self.m) 
                            for yi in xrange(self.m)),
                            dtype=np.float,
                            count=n * self.m * self.m)
        self.gis = self.gis.reshape((n,self.m,self.m))                  
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
        self.alphas = np.fromiter((self.calcalpha(k,v)
                                    for k in xrange(n)
                                    for v in xrange(self.m)),
                                    dtype=np.float,
                                    count=n*self.m
                                )
        self.alphas = self.alphas.reshape((n,self.m))
        return self.alphas
        

    def calcbeta(self, u, k, n):
        if k==n:#I(u==STOP)
            return tags.tags[u] == tags.tags[tags.stop]
        else:
            bet = self.betas[:,k]
            gi = self.gis[k][u]
            temp = gi + bet
            return self.log_sum_exp(temp)


    
    def calcbetas(self, ws, x, y, n):
        self.betas = np.zeros((self.m,n))
        for k in xrange(n,0,-1):
            for u in xrange(self.m):
                self.betas[u][k-1] = self.calcbeta(u,k,n)
                
#        print self.betas
#        print self.betas.shape
#        
#        self.betas = np.fromiter((self.calcbeta(u,k,n)                            
#                                                        
#                            for u in xrange(self.m)
#                            for k in xrange(n,0,-1)
#                            ),
#                            dtype=np.float,
#                            count=self.m*n
#                        )
#        self.betas = self.betas.reshape((self.m,n))
#        
#        print self.betas
#        print self.betas.shape
        return

    
    def calcZ(self, ws, x, y, n):
        zAlpha = self.alphas[n-1][tags.start]
        zBeta = self.betas[tags.start][0]
        #for k in xrange(n+1):
        self.Z = sum((zAlpha*zBeta) for u in xrange(self.m))
            #print self.Z
        #print zAlpha
        #print self.betas
        #print zAlpha, zBeta
        #print zAlpha - zBeta
        #assert zAlpha == zBeta    
        return self.Z
        
        
    def calcF(self, j, x, y, n):
        #return sum(ffs.featureFunc[j](y[i-1], y[i], x, i, n) for i in xrange(n))
        return sum(self.As[j,i] * self.Bs[j,y[i-1],y[i]] for i in xrange(1,n))


    def calcGibbs(self, yi, i, yi1):
        temp = self.gis[i][yi1,:] + self.gis[i][:,yi]
        Gibbs = temp
        summ = self.log_sum_exp(temp)
        P = Gibbs / summ
        M = np.argmax(P)
        self.Gibbs = M#tags.tags[M]
        return self.Gibbs


    def _calcSGDExpect(self, ws, x, y, n):
        
        #calculate forward(alpha) & backward(beta) vectors, and Z
        self.calcalphas(ws, x, y, n)
        self.calcbetas(ws, x, y, n)
        self.Z = self.calcZ(ws, x, y, n)
        
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

    def _calcFExp(self, ws, x, y, n):
        lws=len(ws)
        expect = np.fromiter((self.calcF(j, x, y, n)
                            for j in xrange(lws)),
                            dtype=np.float,
                            count=lws
                        )
        return expect

    def _calcCollExp(self, ws, x, n):
        self.calcUMat(n)
        yhat = self.calcYHat(x)
        return self._calcFExp(ws, x, yhat, n)


    def calcYstar(self, y, n):
        expect = [0]
        for i in xrange(1,n-1):
            yi1 = tags.tags.index(y[i-1])
            yi = tags.tags.index(y[i+1])
            expect.append(self.calcGibbs(yi, i, yi1))
        expect.append(7)
#        ystar = []
#        ystar = (self.calcGibbs(tags.tags.index[y[i+1]], i, tags.tags.index[y[i-1]])
#                            for i in xrange(1,n-1))
#        ystar.insert(0,'START')
#        ystar.append('STOP')
#        print expect,ystar
        return expect

    def _calcCDExpect(self, ws, x, y, n):
        ystar=self.calcYstar(y, n)
        return self._calcFExp(ws, x, ystar, n)


    def calcExpect(self, ws, x, y, n):
        if self.method == "sgd":
            return self._calcSGDExpect(ws, x, y, n)
        elif self.method == "collins":
            return self._calcCollExp(ws, x, n)
        elif self.method == "cd":
            return self._calcCDExpect(ws, x, y, n)
        else:
            print "Incorrect method"
            return -1


    def _sgd_update(self, ws, x, y, rate):
        """single step in SGD"""
        n = len(x)
        
        self.calcAs(x, n)
        self.calcBs()
        #self.calcS(ws, x, n)
        #print len(self.S),self.S
        
        #calculate gi matrices
        self.calcgis(ws, x, n)

        #compute expectation
        yidxs = np.fromiter((tags.tags.index(y[i]) for i in range(n)),
                            dtype=np.float,
                            count=n)
        fval = self._calcFExp(ws, x, yidxs, n)
        expectation = self.calcExpect(ws, x, y, n)        
        
        #p = np.exp(log_prob(1, np.dot(x, ws)))
        result = ws + rate * (fval - expectation)#- 2 * self.mu * ws)
        # do not regularize intercept
        #result[0] = ws[0] + rate * ((y - p) * x[0])
        return result

    @staticmethod
    def tagAccuracy(labels, preds):
        scores = np.fromiter((accuracy_score(label, pred)
                            for label,pred in zip(labels,preds)),
                            dtype=np.float,
                            count=len(preds)
                            )
        return scores
        

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
            tagscores = self.tagAccuracy(labels_valid, prediction)
            score = np.mean(tagscores)
            print score,max(tagscores),min(tagscores)#,self.ws
            if score > 0 and score < old_score:#np.abs(score - old_score) < 1e-8:
                self.converged_ = True
                break
            rate = rate * self.decay
            old_score = score
        if self.converged_:
            print "converged after {} epochs".format(epoch)
        else:
            print "did not converge"
            
        return self.ws

