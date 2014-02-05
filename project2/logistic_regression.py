from __future__ import division

import numpy as np
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

    def __init__(self, method="sgd", mu=0.1, rate=0.1, decay=0.6,
                 max_iters=1000, random_state=None):
        self.method = method
        self.mu = mu
        self.rate = rate
        self.decay = decay
        self.max_iters = 1000
        self.random_state = random_state

    def _validate_args(self):
        methods = ("sgd", "lbfgs")
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

        X = self._preprocess_data(X)
        labels, old_labels = self._preprocess_labels(labels)
        self.old_labels_ = old_labels

        # normalize data
        self.means_ = X.mean(axis=0)
        self.stds_ = X.std(axis=0, ddof=1)
        X = self._normalize_data(X)

        # add intercept
        ones = np.ones(X.shape[0]).reshape(-1, 1)
        X = np.hstack((ones, X))

        if self.method == "sgd":
            betas = self._sgd(X, labels)
        else:
            betas = self._lbfgs(X, labels)

        self.intercept_ = betas[0]
        self.coefficients_ = betas[1:]

    def predict(self, X):
        self._validate_args()
        X = self._preprocess_data(X)
        X = self._normalize_data(X)
        XB = self.intercept_ + np.dot(X, self.coefficients_)
        XB = XB.reshape(-1, 1)
        f = lambda z: log_prob(1, z)
        probs = np.exp(np.apply_along_axis(f, axis=1, arr=XB))
        neg, pos = self.old_labels_
        return np.where(probs.ravel() >= 0.5, pos, neg)

    @staticmethod
    def _preprocess_data(data):
        data = np.array(data)
        assert data.ndim == 2
        return data

    def _normalize_data(self, data):
        return (data - self.means_) / self.stds_

    @staticmethod
    def _preprocess_labels(labels):
        labels = np.array(labels)
        assert labels.ndim <= 2
        if labels.ndim == 2:
            x, y = labels.shape
            assert x == 1 or y == 1
            labels = labels.ravel()
        assert labels.ndim == 1

        all_labels = set(labels)
        assert len(all_labels) == 2

        neg, pos = sorted(list(all_labels))
        labels = np.where(labels == pos, 1, 0)
        return labels, (neg, pos)

    def _sgd_update(self, betas, x, y, rate):
        """single step in SGD"""
        p = np.exp(log_prob(1, np.dot(x, betas)))
        result = betas + rate * ((y - p) * x - 2 * self.mu * betas)
        # do not regularize intercept
        result[0] = betas[0] + rate * ((y - p) * x[0])
        return result

    def _sgd(self, data, labels):
        # shuffle data
        n, k = data.shape
        idx = np.arange(n)
        state = check_random_state(self.random_state)
        state.shuffle(idx)
        data = data[idx]
        labels = labels[idx]

        betas = np.zeros(k)
        rate = self.rate
        self.converged_ = False
        for epoch in range(self.max_iters):
            old_lcl = rlcl(data, labels, betas, self.mu)
            for i, (x, y) in enumerate(zip(data, labels)):
                betas = self._sgd_update(betas, x, y, rate)
            new_lcl = rlcl(data, labels, betas, self.mu)
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
