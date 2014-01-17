from __future__ import division

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
    data = add_intercept(normalize(data))
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

def sigmoid(z):
    # FIXME: fix over/underflow
    return 1 / (1 + np.exp(-z))


def prob(y, x, betas):
    """prob(Y=y | X=x, betas)"""
    p = sigmoid((x * betas).sum())
    if y == 0:
        p = 1 - p
    if p == 0:
        p = 1e-9
    return p


###############################
# stochastic gradient descent #
###############################

def update(betas, x, y, lambda_, mu):
    """single lambda_ in SGD"""
    # TODO: skip sparse calculations
    p = prob(1, x, betas)
    result = betas + lambda_ * ((y - p) * x - 2 * mu * betas)
    # do not regularize intercept
    result[0] = betas[0] + lambda_ * ((y - p) * x[0])
    return result


def lr_sgd(data, labels, mu=1, alpha=1):
    """Logistic regression via SGD

    mu: float
        regularization coefficient

    t0: float
        step size of SGD is calculated as ``mu / (t0 + epoch)``.

    """
    # FIXME: needs better learning rate schedule
    data = preprocess_data(data)
    labels = preprocess_labels(labels)

    n, k = data.shape
    # shuffle data
    idx = np.arange(n)
    np.random.shuffle(idx)
    data = data[idx]
    labels = labels[idx]

    # initial guess
    betas = np.zeros(k)

    # iterate until convergence
    converged = False
    epoch = 1
    while not converged:
        old = betas.copy()
        lambda_ = alpha / epoch
        for x, y in zip(data, labels):
            # update betas
            betas = update(betas, x, y, lambda_, mu)
        # update step size
        epoch += 1
        converged = np.linalg.norm(betas - old, ord=2) < 1e-5
    return betas


##########
# L-BFGS #
##########

def lcl(data, labels, betas):
    """log conditional likelihood"""
    return sum(np.log(prob(y, x, betas)) for x, y in zip(data, labels))


def rlcl(data, labels, betas, mu):
    """regularized log conditional likelihood"""
    return lcl(data, labels, betas) - mu * np.linalg.norm(betas[1:], ord=2)


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


def lr_lbfgs(data, labels, mu=1):
    """logistic regression using L-BFGS"""
    data = preprocess_data(data)
    labels = preprocess_labels(labels)

    f = lambda b: rlcl(data, labels, b, mu)
    fprime = lambda b: rlcl_prime(data, labels, b, mu)

    x0 = np.zeros(data.shape[1])

    result = fmin_l_bfgs_b(f, x0, fprime)
    return result[0]


##############
# prediction #
##############

def predict(data, betas):
    """Predict class labels of a new data set."""
    data = preprocess_data(data)
    result = sigmoid(data.dot(betas.reshape(-1, 1)))
    return np.where(result.ravel() >= 0.5, 1, 0)
