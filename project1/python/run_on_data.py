import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from logistic_regression import LogisticRegression


def process_line(line):
    line = line.strip()
    elts = line.split(' ')
    label = int(elts[0])
    pairs = list(e.split(':') for e in elts[1:])
    idx, vals = zip(*pairs)
    sample = np.array(map(int, vals))
    return sample, label


def read_file(filename):
    """Read a data file and return data matrix and labels"""
    f = open(filename)
    lines = f.read().split('\n')
    pairs = list(process_line(i) for i in lines if len(i.strip()) > 0)
    f.close()
    samples, labels = zip(*pairs)
    samples = np.array(samples)
    labels = np.array(labels)
    return samples, labels


def sgd(mus, rates, decays, data, labels, data_train, labels_train,
        data_valid, labels_valid, data_test, labels_test):
    print "starting grid search for SGD"
    validation_results = {}
    for mu in mus:
        for rate in rates:
            for decay in decays:
                print "trying mu={} rate={} decay={}".format(mu, rate, decay)
                model = LogisticRegression(method="sgd", mu=mu,
                                           rate=rate, decay=decay)
                model.fit(data_train, labels_train)
                prediction = model.predict(data_valid)
                score = accuracy_score(labels_valid, prediction)
                validation_results[(mu, rate, decay)] = score
                print "  score: {}".format(score)
                print "  error rate: {}".format(1 - score)

    print "evaluating on test set"
    # get hyperparameters for highest accuracy on validation set
    mu, rate, decay = max(validation_results, key=validation_results.get)
    print "Using mu={} rate={} decay={}".format(mu, rate, decay)

    # train on entire train set and predict on test set
    model = LogisticRegression(method="sgd", mu=mu, rate=rate, decay=decay)
    model.fit(data, labels)
    prediction = model.predict(data_test)
    sgd_score = accuracy_score(labels_test, prediction)

    print "SGD test score: {}, error rate: {}".format(sgd_score, 1 - sgd_score)


def lbfgs(mus, data, labels, data_train, labels_train,
          data_valid, labels_valid, data_test, labels_test):
    print "starting grid search for L-BFGS"
    validation_results = {}
    for mu in mus:
        print "trying mu={}".format(mu)
        model = LogisticRegression(method="lbfgs", mu=mu)
        model.fit(data_train, labels_train)
        prediction = model.predict(data_valid)
        score = accuracy_score(labels_valid, prediction)
        validation_results[mu] = score
        print "  score: {}".format(score)
        print "  error rate: {}".format(1 - score)

    print "evaluating on test set"

    # get hyperparameters for highest accuracy on validation set
    mu = max(validation_results, key=validation_results.get)

    print "Using mu of {}".format(mu)

    # train on entire train set and predict on test set
    model = LogisticRegression(method="lbfgs", mu=mu)
    model.fit(data, labels)
    prediction = model.predict(data_test)
    score = accuracy_score(labels_test, prediction)

    print "L-BFGS test score: {}, error rate: {}".format(score, 1 - score)


if __name__ == "__main__":
    # read data and split training data into training and validation sets
    data, labels = read_file('../1571/train.txt')
    data_train, data_valid, labels_train, labels_valid = \
        train_test_split(data, labels, test_size=0.3)

    data_test, labels_test = read_file('../1571/test.txt')

    # hyperparameters to try
    mus = list(10 ** x for x in range(-4, 1))
    rates = list(10 ** x for x in range(-3, 1))
    decays = [0.3, 0.6, 0.9]

    sgd(mus, rates, decays, data, labels, data_train, labels_train,
        data_valid, labels_valid, data_test, labels_test)

    lbfgs(mus, data, labels, data_train, labels_train,
          data_valid, labels_valid, data_test, labels_test)
