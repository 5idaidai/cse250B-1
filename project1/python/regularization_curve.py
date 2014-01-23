from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as pp

from util import read_file
from logistic_regression import LogisticRegression


data, labels = read_file('../1571/train.txt')
data_train, data_valid, labels_train, labels_valid = \
    train_test_split(data, labels, test_size=0.3, random_state=0)

mus = list(10 ** x for x in range(-8, 2))

sgd_scores = []
for mu in mus:
    sgd_model = LogisticRegression(method="sgd", mu=mu, rate=0.1,
                                   decay=0.6, random_state=0)
    sgd_model.fit(data_train, labels_train)
    predicted = sgd_model.predict(data_valid)
    sgd_scores.append(accuracy_score(labels_valid, predicted))

pp.figure()
pp.xscale('log')
pp.scatter(mus, sgd_scores)
pp.xlabel('regularization strength')
pp.ylabel('accuracy')
pp.savefig('./sgd_regularization.png')


lbfgs_scores = []
for mu in mus:
    sgd_model = LogisticRegression(method="lbfgs", mu=mu, rate=0.1,
                                   decay=0.6, random_state=0)
    sgd_model.fit(data_train, labels_train)
    predicted = sgd_model.predict(data_valid)
    lbfgs_scores.append(accuracy_score(labels_valid, predicted))

pp.figure()
pp.xscale('log')
pp.scatter(mus, lbfgs_scores)
pp.xlabel('regularization strength')
pp.ylabel('accuracy')
pp.savefig('./lbfgs_regularization.png')
