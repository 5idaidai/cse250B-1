import unittest

import numpy as np

from scipy.optimize import check_grad

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

import logistic_regression as lr


class TestPrimes(unittest.TestCase):

    def setUp(self):
        n_samples = 100
        n_features = 3
        n_betas = 100
        data, labels = make_blobs(n_samples, n_features, centers=2,
                                  random_state=0)
        self.data = lr.LogisticRegression._preprocess_data(data)
        self.labels, _ = lr.LogisticRegression._preprocess_labels(labels)
        state = np.random.RandomState(0)
        self.betas = state.uniform(-10, 10, size=(n_betas, data.shape[1]))

    def _test(self, f, fprime):
        for b in self.betas:
            ret = check_grad(f, fprime, b)
            assert abs(ret) < 1e-2

    def test_lcl(self):
        f = lambda b: lr.lcl(self.data, self.labels, b)
        fprime = lambda b: lr.lcl_prime(self.data, self.labels, b)
        self._test(f, fprime)

    def test_rlcl(self):
        f = lambda b: lr.rlcl(self.data, self.labels, b, mu=1)
        fprime = lambda b: lr.rlcl_prime(self.data, self.labels, b, mu=1)
        self._test(f, fprime)


class TestLogisticRegression(unittest.TestCase):


    def test_preprocess_labels(self):
        labels = [-1, 1, -1]
        expected = np.array([0, 1, 0])
        result, old = lr.LogisticRegression._preprocess_labels(labels)
        assert np.all(expected == result)
        assert old == (-1, 1)

    def test_predicted_labels(self):
        data = np.array([0, 0, 1, 1]).reshape([-1, 1])
        labels = np.array([-1, -1, 1, 1])
        model = lr.LogisticRegression(random_state=0)
        model.fit(data, labels)
        predictions = model.predict(data)
        self.assertTrue(np.all(predictions == labels))

    def test_sgd(self):
        data, labels = make_blobs(n_features=5, centers=2)
        data_a, data_b, labels_a, labels_b = train_test_split(data, labels)
        model = lr.LogisticRegression(method="sgd", random_state=0)
        model.fit(data_a, labels_a)
        predictions = model.predict(data_b)
        score = accuracy_score(labels_b, predictions)
        self.assertTrue(score > 0.9)

    def test_lbfgs(self):
        data, labels = make_blobs(n_features=5, centers=2)
        data_a, data_b, labels_a, labels_b = train_test_split(data, labels)
        model = lr.LogisticRegression(method="lbfgs", random_state=0)
        model.fit(data, labels)
        model.fit(data_a, labels_a)
        predictions = model.predict(data_b)
        score = accuracy_score(labels_b, predictions)
        self.assertTrue(score > 0.9)


if __name__ == "__main__":
    unittest.main()
