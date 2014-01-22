from __future__ import division

import collections
import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def _sigmoid(z):
    if z > 36:
        return 1 - 1e-9
    elif z < -709:
        return 1e-9
    else:
        return 1 / (1 + np.exp(-z))


def sigmoid(z):
    if isinstance(z, collections.Iterable):
        return np.array(list(_sigmoid(n) for n in np.nditer(z)))
    else:
        return _sigmoid(z)


def prob(y, x, betas):
    """prob(Y=y | X=x, betas)"""
    p = sigmoid((x * betas).sum())
    if y == 0:
        p = 1 - p
    if p == 0:  # TODO: this is a hack
        raise Exception('this should never happen')
        p = 1e-9
    return p


def lcl(data, labels, betas):
    """log conditional likelihood"""
    return sum(np.log(prob(y, x, betas)) for x, y in zip(data, labels))


def rlcl(data, labels, betas, mu):
    """regularized log conditional likelihood"""
    #return lcl(data, labels, betas) - mu * np.linalg.norm(betas[1:], ord=2)
    return lcl(data, labels, betas) - mu * sum(np.power(betas[1:], 2))


def lcl_prime(data, labels, betas):
    """gradient of lcl"""
    coeffs = np.array(list(y - prob(1, x, betas)
                           for x, y in zip(data, labels)))
    return coeffs.dot(data)


def rlcl_prime(data, labels, betas, mu):
    """gradient of rlcl"""
    grad = lcl_prime(data, labels, betas)
    result = grad - 2 * mu * betas
    result[0] = grad[0]  # do not regularize intercept
    return result


class LogisticRegression(object):
    """Logistic regression model with L2 regularization..

    Parameters
    ----------
    method: string
        May be "sgd" or "lbfgs".

    mu: float
        Strength of regularization.

    alpha:
        Step size of SGD. Calculated as ``alpha / (epoch + 1)``.

    max_iters:
        Maximum iterations of SGD.

    """

    def __init__(self, method="sgd", mu=1, alpha=1, max_iters=1000):
        self.method = method
        self.mu = mu
        self.alpha = alpha
        self.max_iters = 1000

    def _validate_args(self):
        methods = ("sgd", "lbfgs")
        if self.method not in methods:
            raise Exception("method '{}' invalid. should be"
                            " one of {}".format(self.method, methods))
        if self.mu <= 0:
            raise Exception("invalid regularization strength. mu={},"
                            " but it should be > 0".format(self.mu))

        if self.alpha <= 0:
            raise Exception("invalid step schedule. alpha={},"
                            " but it should be > 0".format(self.alpha))

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
        z = self.intercept_ + np.dot(X, self.coefficients_)
        probs = sigmoid(z)
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

    def _sgd_update(self, betas, x, y, lambda_):
        """single step in SGD"""
        p = prob(1, x, betas)
        result = betas + lambda_ * ((y - p) * x - 2 * self.mu * betas)
        # do not regularize intercept
        result[0] = betas[0] + lambda_ * ((y - p) * x[0])
        return result

    def _sgd(self, data, labels):
        # FIXME: needs better learning rate schedule
        # TODO: check for convergence during epoch

        # shuffle data
        n, k = data.shape
        idx = np.arange(n)
        np.random.shuffle(idx)
        data = data[idx]
        labels = labels[idx]

        betas = np.zeros(k)
        lambda_ = self.alpha
        self.converged_ = False
        for epoch in range(self.max_iters):
            old_lcl = rlcl(data, labels, betas, self.mu)
            for i, (x, y) in enumerate(zip(data, labels)):
                betas = self._sgd_update(betas, x, y, lambda_)
            new_lcl = rlcl(data, labels, betas, self.mu)
            if np.abs(new_lcl - old_lcl) < 1e-8:
                self.converged_ = True
                break
            lambda_ = lambda_ * 0.6
        return betas

    def _lbfgs(self, data, labels):
        f = lambda b: -rlcl(data, labels, b, self.mu)
        fprime = lambda b: -rlcl_prime(data, labels, b, self.mu)
        x0 = np.zeros(data.shape[1])
        result = fmin_l_bfgs_b(f, x0, fprime)
        print result[2]['warnflag'], result[2]['task']
        return result[0]
