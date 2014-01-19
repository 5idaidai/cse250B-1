import numpy as np
from scipy.optimize import check_grad

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

import logistic_regression as lr

def test_rlcl():
    data = [[1, 0, 1],
            [1, 1, 1]]
    labels = [0, 1]
    betas = np.array([1, 1, 1])
    mu = 2
    result = lr.rlcl(data, labels, betas, mu)
    #expected = (np.log(1 - lr.sigmoid(2)) + np.log(lr.sigmoid(3)) -
    #            mu * np.sqrt(2))
    expected = (np.log(1 - lr.sigmoid(2)) + np.log(lr.sigmoid(3)) -
                mu * 2)                
    assert result == expected


def test_primes():
    data, labels = make_blobs(centers=2)
    data = lr.preprocess_data(data)
    labels = lr.preprocess_labels(labels)
    betas = np.random.rand(100, data.shape[1])

    f = lambda b: lr.lcl(data, labels, b)
    fprime = lambda b: lr.lcl_prime(data, labels, b)
    for bs in betas:
        assert abs(check_grad(f, fprime, bs)) < 0.1

    f = lambda b: lr.rlcl(data, labels, b, mu=1)
    fprime = lambda b: lr.rlcl_prime(data, labels, b, mu=1)
    for bs in betas:
        check_grad(f, fprime, bs)
        assert abs(check_grad(f, fprime, bs)) < 0.1
        

def test_logistic_regression():
    data, labels = make_blobs(n_features=5, centers=2)

    betas = lr.lr_sgd(data, labels)
    predictions = lr.predict(data, betas)
    score = accuracy_score(labels, predictions)
    assert score > 0.9

    betas = lr.lr_lbfgs(data, labels)
    predictions = lr.predict(data, betas)
    score = accuracy_score(labels, predictions)
    assert score > 0.9

	
#testarr = range(6)
#print sum(np.power(testarr,2))
#print np.linalg.norm(testarr, ord=2)

test_rlcl()
test_primes()
test_logistic_regression()