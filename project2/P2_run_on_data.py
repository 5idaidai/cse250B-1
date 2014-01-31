from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from util import read_file


"""Data processing:

1. Determine method of distinguishing tags from words for all word-tag pairs
2. Determine delimeters for separating each word in each sentence
3. Parse all word-tag pairs into a 3-dimensional matrix with word-tag-sentence(sentence=label) axes??
4. Split training data into train and validation sets
"""

def importData():
    # read data and split training data into training and validation sets
    data, labels = read_file('training')
    data_train, data_valid, labels_train, labels_valid = \
        train_test_split(data, labels, test_size=0.3, random_state=0)

    test_data, test_labels = read_file('test')
    return data_train, data_valid, test_data, labels_train, labels_valid, test_labels
    

"""Feature Functions:

1. Write script to generate FFs for all sentences based on A and B functions
2. Determine what the A and B functions should be, and how many we should use
3. Generate AxB FFs for all training data using script
4. FF output is {1,0} where 0 is the majority of outputs
5. Create set S that is only non-zero outputs for all FFs
"""

"""Training:

1. Maximize LCL
2. Regularization??
3. Update wj using wj + lambda[Fj(x,y)-Expectation[Fj(x,y')]] 
4. Collins Perceptron
5. L-BFGS
"""

"""Prediction:

1. Predict y-hat based on S for each sentence(x-bar)
"""

"""Check functions:

1. Use forward and backward vectors to check that Z(x-bar,w) is the same for both
2. Checkgrad
3. Check data processing??
4. Check FFs??
"""

if __name__ == "__main__":
    data_train, data_valid, data_test, labels_train, labels_valid, labels_test = importData()
    