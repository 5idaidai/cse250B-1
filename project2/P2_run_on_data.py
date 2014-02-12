from sklearn.metrics import accuracy_score
from util import read_file
from logistic_regression import LogisticRegression
from datetime import datetime

"""Data processing:

1. Determine method of distinguishing tags from words for all word-tag pairs
2. Determine delimeters for separating each word in each sentence
3. Parse all word-tag pairs into a 3-dimensional matrix with word-tag-sentence(sentence=label) axes??
4. Split training data into train and validation sets
"""

def importData():
    # read data and split training data into training and validation sets
    data_train, labels_train = read_file('medium')
        
    #assert len(data_train[0]) == len(labels_train[0])
    #assert len(data_train[200]) == len(labels_train[200])

    data_test, labels_test = read_file('short')
            
    #assert len(data_test[0]) == len(data_test[0])
    #assert len(data_test[200]) == len(data_test[200])
    
    return data_train, data_test, labels_train, labels_test
    
def runML(meth, itrs, data_train, data_test, labels_train, labels_test):
    print meth,datetime.now().time()
    model = LogisticRegression(method=meth,max_iters=itrs)
    model.fit(data_train, labels_train)
    print datetime.now().time()
    prediction = model.predict(data_test)
    score = accuracy_score(labels_test, prediction)
    print "  score: {}".format(score)
    print "  error rate: {}".format(1 - score)
    print datetime.now().time()

if __name__ == "__main__":
    data_train, data_test, labels_train, labels_test = importData()
    labels_test=LogisticRegression.preproclabels(labels_test)
    
    runML("collins",10,data_train, data_test, labels_train, labels_test)
    #runML("cd",10,data_train, data_test, labels_train, labels_test)