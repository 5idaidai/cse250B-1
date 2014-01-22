import numpy as np
from scipy.optimize import check_grad

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

import logistic_regression as lr

def test_sigmoid():
    print "Testing sigmoid"
    data = np.array(range(-50000,0,10))
    data2 = np.array(range(0,50000,10))
    
    negs = lr.sigmoid(data)
    pos = lr.sigmoid(data2)
    
    assert max(negs) <= 0.5, "Wrong negs"
    assert min(negs) > 0, "Wrong negs, less than 0"
    assert min(pos) >= 0.5, "Wrong pos"
    assert max(pos) < 1, "Wrong pos, greater than 1"

    print "Sigmoid passed"


def test_process_labels():
    labels = [-1, 1, -1]
    expected = np.array([0, 1, 0])
    result = lr.preprocess_labels(labels)
    assert expected == result


def test_rlcl():
    print "Testing RLCL"
    data = [[40000, 0, 1],
            [1, 1, -40000]]
    labels = [0, 1]
    betas = np.array([1, 1, 1])
    mu = 2
    result = lr.rlcl(data, labels, betas, mu)
    #expected = (np.log(1 - lr.sigmoid(40001)) + np.log(lr.sigmoid(-39998)) -
    #            mu * np.sqrt(2))
    s1 = lr.sigmoid(40001)
    s2 = lr.sigmoid(-39998)
    expected = (np.log(1 - s1) + np.log(s2) -
                mu * 2)                
    if result == expected:
        print "rlcl: Pass"
    else:
        print "rlcl: Fail, result: {}, expected: {}".format(result,expected)


def test_primes():
    print "Testing gradients"
    data, labels = make_blobs(centers=2)
    data = lr.preprocess_data(data)
    labels = lr.preprocess_labels(labels)
    betas = np.random.rand(100, data.shape[1])
    betas_neg = np.random.rand(100, data.shape[1]) * -1

    print "Checking LCL"
    f = lambda b: lr.lcl(data, labels, b)
    fprime = lambda b: lr.lcl_prime(data, labels, b)
    for bs in betas:
        ret = check_grad(f, fprime, bs)
        assert abs(ret) < 0.1, "Fail {}".format(ret)
    for bs in betas_neg:
        ret = check_grad(f, fprime, bs)
        assert abs(ret) < 0.1, "Fail {}".format(ret)
    print "Passed"

    print "Checking RLCL"
    f = lambda b: lr.rlcl(data, labels, b, mu=1)
    fprime = lambda b: lr.rlcl_prime(data, labels, b, mu=1)
    for bs in betas:
        ret = check_grad(f, fprime, bs)
        assert abs(ret) < 0.1, "Fail {}".format(ret)
    for bs in betas_neg:
        ret = check_grad(f, fprime, bs)
        assert abs(ret) < 0.1, "Fail {}".format(ret)        
    print "Passed"
        
def test_lr():
    test_lr_sgd()
    test_lr_lbfgs()

def test_lr_sgd():
    data, labels = make_blobs(n_features=5, centers=2)

    betas = lr.lr_sgd(data, labels, 1, 0.2)
    predictions = lr.predict(data, betas)

    score = accuracy_score(labels, predictions)
    if score > 0.9:
        print "lr_sgd: Pass {}".format(score)
    else:
        print "lr_sgd: Fail score={}".format(score)
          
def test_lr_lbfgs():
    data, labels = make_blobs(n_features=5, centers=2)

    betas = lr.lr_lbfgs(data, labels, 0.5)
    predictions = lr.predict(data, betas)
    score = accuracy_score(labels, predictions)
    if score > 0.9:
        print "lr_lbfgs: Pass score={}".format(score)
    else:
        print "lr_lbfgs: Fail score={}".format(score)

test_sigmoid()
test_rlcl()
test_primes()
test_lr()
