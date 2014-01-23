from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

from logistic_regression import LogisticRegression
from util import read_file


def sgd(mus, rates, decays, data, labels, data_train, labels_train,
        data_valid, labels_valid, data_test, labels_test):
    print "starting grid search for SGD"
    validation_results = {}
    dicts = []
    for mu in mus:
        for rate in rates:
            for decay in decays:
                print "trying mu={} rate={} decay={}".format(mu, rate, decay)
                model = LogisticRegression(method="sgd", mu=mu,
                                           rate=rate, decay=decay,
                                           random_state=0)
                model.fit(data_train, labels_train)
                prediction = model.predict(data_valid)
                score = accuracy_score(labels_valid, prediction)
                validation_results[(mu, rate, decay)] = score
                print "  score: {}".format(score)
                print "  error rate: {}".format(1 - score)

                d = dict(method="sgd", mu=mu, rate=rate, decay=decay,
                         score=score, lcl=model.lcl_,
                         rlcl=model.rlcl_, test=False)
                dicts.append(d)

    print "evaluating on test set"
    # get hyperparameters for highest accuracy on validation set
    mu, rate, decay = max(validation_results, key=validation_results.get)
    print "Using mu={} rate={} decay={}".format(mu, rate, decay)

    # train on entire train set and predict on test set
    model = LogisticRegression(method="sgd", mu=mu, rate=rate,
                               decay=decay, random_state=0)
    model.fit(data, labels)
    prediction = model.predict(data_test)
    score = accuracy_score(labels_test, prediction)

    print "SGD test score: {}, error rate: {}".format(score, 1 - score)

    d = dict(method="sgd", mu=mu, rate=rate, decay=decay, score=score,
             lcl=model.lcl_, rlcl=model.rlcl_, test=True)
    dicts.append(d)
    return pd.DataFrame(dicts)


def lbfgs(mus, data, labels, data_train, labels_train,
          data_valid, labels_valid, data_test, labels_test):
    print "starting grid search for L-BFGS"
    validation_results = {}
    dicts = []
    for mu in mus:
        print "trying mu={}".format(mu)
        model = LogisticRegression(method="lbfgs", mu=mu)
        model.fit(data_train, labels_train)
        prediction = model.predict(data_valid)
        score = accuracy_score(labels_valid, prediction)
        validation_results[mu] = score
        print "  score: {}".format(score)
        print "  error rate: {}".format(1 - score)

        d = dict(method="lbfgs", mu=mu, rate=-1, decay=-1,
                 score=score, lcl=model.lcl_, rlcl=model.rlcl_,
                 test=False)
        dicts.append(d)

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

    d = dict(method="lbfgs", mu=mu, rate=-1, decay=-1,
             score=score, lcl=model.lcl_, rlcl=model.rlcl_, test=True)
    dicts.append(d)
    return pd.DataFrame(dicts)


if __name__ == "__main__":
    # read data and split training data into training and validation sets
    data, labels = read_file('../1571/train.txt')
    data_train, data_valid, labels_train, labels_valid = \
        train_test_split(data, labels, test_size=0.3, random_state=0)

    data_test, labels_test = read_file('../1571/test.txt')

    # hyperparameters to try
    mus = list(10 ** x for x in range(-4, 1))
    rates = list(10 ** x for x in range(-3, 1))
    decays = [0.3, 0.6, 0.9]

    sgd_df = sgd(mus, rates, decays, data, labels, data_train,
                 labels_train, data_valid, labels_valid, data_test,
                 labels_test)

    lbfgs_df = lbfgs(mus, data, labels, data_train, labels_train,
                     data_valid, labels_valid, data_test, labels_test)

    df = pd.concat((sgd_df, lbfgs_df))
    cols = ["method", "mu", "rate", "decay", "score", "lcl", "rlcl", "test"]
    df = df[cols]
    df.to_csv('./results.csv', index=False)
