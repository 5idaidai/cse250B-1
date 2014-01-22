import unittest

import numpy as np
from scipy.optimize import check_grad

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

import logistic_regression as lr


class TestPrimes(unittest.TestCase):

    def setUp(self):
        n_samples = 100
        n_features = 3
        data, labels = make_blobs(n_samples, n_features, centers=2)
        self.data = lr.LogisticRegression._preprocess_data(data)
        self.labels, _ = lr.LogisticRegression._preprocess_labels(labels)
        self.betas = np.random.rand(100, data.shape[1])
        self.betas_neg = np.random.rand(100, data.shape[1]) * -1

    def test_lcl(self):
        f = lambda b: lr.lcl(self.data, self.labels, b)
        fprime = lambda b: lr.lcl_prime(self.data, self.labels, b)
        for bs in self.betas:
            ret = check_grad(f, fprime, bs)
            assert abs(ret) < 0.1
        for bs in self.betas_neg:
            ret = check_grad(f, fprime, bs)
            assert abs(ret) < 0.1

    def test_rlcl(self):
        f = lambda b: lr.rlcl(self.data, self.labels, b, mu=1)
        fprime = lambda b: lr.rlcl_prime(self.data, self.labels, b, mu=1)
        for bs in self.betas:
            ret = check_grad(f, fprime, bs)
            assert abs(ret) < 0.1
        for bs in self.betas_neg:
            ret = check_grad(f, fprime, bs)
            assert abs(ret) < 0.1


class TestLogisticRegression(unittest.TestCase):

    def test_sigmoid(self):
        data = np.array(range(-50000, 0, 10))
        data2 = np.array(range(0, 50000, 10))

        negs = lr.sigmoid(data)
        pos = lr.sigmoid(data2)

        assert max(negs) <= 0.5
        assert min(negs) > 0
        assert min(pos) >= 0.5
        assert max(pos) < 1

    def test_preprocess_labels(self):
        labels = [-1, 1, -1]
        expected = np.array([0, 1, 0])
        result, old = lr.LogisticRegression._preprocess_labels(labels)
        assert np.all(expected == result)
        assert old == (-1, 1)

    def test_rlcl(self):
        data = [[40000, 0, 1],
                [1, 1, -40000]]
        labels = [0, 1]
        betas = np.array([1, 1, 1])
        mu = 2
        result = lr.rlcl(data, labels, betas, mu)
        s1 = lr.sigmoid(40001)
        s2 = lr.sigmoid(-39998)
        expected = (np.log(1 - s1) + np.log(s2) -
                    mu * 2)
        self.assertEquals(result, expected)

    def test_sgd(self):
        data, labels = make_blobs(n_features=5, centers=2)

        # also ensure relabeling is working
        labels = np.array(labels)
        labels = np.where(labels == 1, 2, -2)

        model = lr.LogisticRegression(method="sgd")
        model.fit(data, labels)
        predictions = model.predict(data)
        score = accuracy_score(labels, predictions)
        self.assertTrue(score > 0.9)

    def test_lbfgs(self):
        data, labels = make_blobs(n_features=5, centers=2)
        model = lr.LogisticRegression(method="lbfgs")
        model.fit(data, labels)
        predictions = model.predict(data)
        score = accuracy_score(labels, predictions)
        self.assertTrue(score > 0.9)


if __name__ == "__main__":
    unittest.main()
