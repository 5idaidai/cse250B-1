from __future__ import division

import collections
import numpy as np
from scipy.optimize import fmin_l_bfgs_b


##################
# data functions #
##################

def add_intercept(data):
    """Adds an extra dimension of ones."""
    n = data.shape[0]
    ones = np.ones(n).reshape(-1, 1)
    return np.hstack((ones, data))


def normalize(data):
    """Normalize each feature to mean=1 and std=0"""
    return (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)


def preprocess_data(data):
    """Normalize and add intercept."""
    data = np.array(data)
    assert data.ndim == 2
    data = add_intercept(data)
    return data


def preprocess_labels(labels):
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
    return labels


####################
# common functions #
####################

def _sigmoid(z):
    if z > 36:
        return 1 - 1e-9
    elif z < -709:
        return 1e-9
    else:
        return 1 / (1+np.exp(-z))
        

def sigmoid(z):
    if not isinstance(z,collections.Iterable):
        return _sigmoid(z)
    else:
        return np.array(list(_sigmoid(n) for n in np.nditer(z)))


def prob(y, x, betas):
    """prob(Y=y | X=x, betas)"""
    p = sigmoid((x * betas).sum())
    if y == 0:
        p = 1 - p
    if p == 0: # TODO: this is a hack
        raise Exception('this should never happen')
        p = 1e-9
    return p


###############################
# stochastic gradient descent #
###############################

def update(betas, x, y, lambda_, mu):
    """single step in SGD"""
    # TODO: skip sparse calculations
    p = prob(1, x, betas)
    result = betas + lambda_ * ((y - p) * x - 2 * mu * betas)
    # do not regularize intercept
    result[0] = betas[0] + lambda_ * ((y - p) * x[0])
    return result


def lr_sgd(data, labels, mu=1, alpha=1, max_iters=1000):
    """Logistic regression via SGD

mu: float
Regularization coefficient.

alpha: float
Step size of SGD is calculated as ``alpha / (epoch + 1)``.

"""
    # FIXME: needs better learning rate schedule
    # TODO: check for convergence during epoch

    data = preprocess_data(data)
    labels = preprocess_labels(labels)

    # shuffle data
    n, k = data.shape
    idx = np.arange(n)
    np.random.shuffle(idx)
    data = data[idx]
    labels = labels[idx]

    betas = np.zeros(k)
    for epoch in range(max_iters):
        old = betas.copy()
        old_lcl = rlcl(data, labels, betas, mu)
        lambda_ = alpha #/ (epoch + 1)
        for x, y in zip(data, labels):
            betas = update(betas, x, y, lambda_, mu)
        new_lcl = rlcl(data, labels, betas, mu)
        print old_lcl, new_lcl
        if np.abs(new_lcl - old_lcl) < 1e-4:
            print "Converged at epoch", epoch
            break # converged
    return betas


##########
# L-BFGS #
##########

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
    result[0] = grad[0] # do not regularize intercept
    return result


def lr_lbfgs(data, labels, mu=1):
    """logistic regression using L-BFGS"""
    data = preprocess_data(data)
    labels = preprocess_labels(labels)

    f = lambda b: rlcl(data, labels, b, mu)
    fprime = lambda b: rlcl_prime(data, labels, b, mu)

    x0 = np.zeros(data.shape[1])

    result = fmin_l_bfgs_b(f, x0, fprime)
    print result[2]['warnflag'], result[2]['task'], result[2]['grad']
    return result[0]
    
    
##############
# prediction #
##############

def predict(data, betas):
    """Predict class labels of a new data set."""
    data = preprocess_data(data)
    result = sigmoid(data.dot(betas.reshape(-1, 1)))
    return np.where(result.ravel() >= 0.5, 1, 0)
