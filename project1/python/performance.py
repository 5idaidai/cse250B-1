from __future__ import division

import timeit
from logistic_regression import LogisticRegression
from util import read_file


data, labels = read_file('../1571/train.txt')
data_test, labels_test = read_file('../1571/test.txt')

sgd_model = LogisticRegression(method="sgd", mu=0.01, rate=0.1,
                               decay=0.6, random_state=0)


def sgd_fit():
    sgd_model.fit(data, labels)


def sgd_predict():
    sgd_model.predict(data_test)

lbfgs_model = LogisticRegression(method="lbfgs", mu=0.001, random_state=0)


def lbfgs_fit():
    lbfgs_model.fit(data, labels)


def lbfgs_predict():
    lbfgs_model.predict(data_test)


if __name__ == '__main__':
    number = 3
    sgd_fit_time = timeit.timeit("sgd_fit()",
                                 setup="from __main__ import sgd_fit",
                                 number=number)
    sgd_predict_time = timeit.timeit("sgd_predict()",
                                     setup="from __main__ import sgd_predict",
                                     number=number)
    lbfgs_fit_time = timeit.timeit("lbfgs_fit()",
                                   setup="from __main__ import lbfgs_fit",
                                   number=number)
    lbfgs_predict_time = timeit.timeit("lbfgs_predict()",
                                       setup="from __main__ import lbfgs_predict",
                                       number=number)

    print "SGD fit        : {}".format(sgd_fit_time / number)
    print "SGD predict    : {}".format(sgd_predict_time / number)
    print "L-BFGS fit     : {}".format(lbfgs_fit_time / number)
    print "L-BFGS predict : {}".format(lbfgs_predict_time / number)
