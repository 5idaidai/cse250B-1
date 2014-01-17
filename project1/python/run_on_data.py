import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

import logistic_regression as lr


def process_line(line):
    line = line.strip()
    elts = line.split(' ')
    label = int(elts[0])
    pairs = list(e.split(':') for e in elts[1:])
    idx, vals = zip(*pairs)
    sample = np.array(map(int, vals))
    return sample, label


def read_file(filename):
    f = open(filename)
    lines = f.read().split('\n')
    pairs = list(process_line(i) for i in lines if len(i.strip()) > 0)
    f.close()
    samples, labels = zip(*pairs)
    samples = np.array(samples)
    labels = np.array(labels)
    return samples, labels


data, labels = read_file('../1571/train')
data_train, data_valid, labels_train, labels_valid = \
    train_test_split(data, labels)

data_test, labels_test = read_file('../1571/test')

mus = list(10 ** x for x in range(-1, 5))
alphas = [1]

results = {}

print "starting grid search"
for mu in mus:
    for alpha in alphas:
        print "trying mu={} alpha={}".format(mu, alpha)
        betas = lr.lr_sgd(data_train, labels_train, mu=mu, alpha=alpha)
        prediction = lr.predict(data_valid, betas)
        score = accuracy_score(labels_valid, prediction)
        results[(mu, alpha)] = score
        print "  score: {}".format(score)

mu, alpha = max(results, key=results.get)
betas = lr.lr_sgd(data_train, labels_train, mu=mu, alpha=alpha)
prediction = lr.predict(data_test, betas)
sgd_score = accuracy_score(labels_valid, prediction)

print "SGD score: {}".format(sgd_score)
