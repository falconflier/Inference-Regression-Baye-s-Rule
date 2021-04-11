import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from housing import stand_n, estimate


# All the code that I've written to classify the data for the titanic (survived or died)
def predict_survival():
    num_features = 6
    num_data = 887
    num_training = int(4 * num_data / 5)
    read = pd.read_csv("titanic.csv")
    data = np.array(read)
    np.random.shuffle(data)
    # stand_n(data, num_training)
    # The design matrix
    X = data[:num_training, :num_features]
    y = data[:num_training, num_features]
    test_feats = data[num_training:, :num_features]
    test_labels = data[num_training:, num_features]
    y = y.astype('int')

    #######################################################################
    # Naive Bayes
    gnb = GaussianNB()
    y_pred = gnb.fit(X, y).predict(test_feats)
    print("Number of mislabeled points out of a total %d points : %d" % (test_feats.shape[0], (test_labels != y_pred).sum()))
    print("Naive Baye's Score: " + str(1 - (test_labels != y_pred).sum() / (test_feats.shape[0])))

    #######################################################################
    # Logistic Regression
    clf = LogisticRegression().fit(X, y)
    clf.predict(test_feats)
    print("Logistic Regression's Score: " + str(clf.score(X, y)))


def spam_filter():
    num_data = 4601
    num_feats = 57
    num_training = int(4 * num_data / 5)
    read = pd.read_csv("spam.csv")
    data = np.array(read)
    np.random.shuffle(data)
    X = data[:num_training, :num_feats]
    y = data[:num_training, num_feats]
    test_feats = data[num_training:, :num_feats]
    test_labels = data[num_training:, num_feats]
    y = y.astype('int')
    test_labels = test_labels.astype('int')

    #######################################################################
    # Naive Bayes
    gnb = GaussianNB()
    y_pred = gnb.fit(X, y).predict(test_feats)
    print("Number of mislabeled points out of a total %d points : %d" % (test_feats.shape[0], (test_labels != y_pred).sum()))
    print("Naive Baye's Score on spam: " + str(1 - (test_labels != y_pred).sum() / (test_feats.shape[0])))

    #######################################################################
    # Logistic Regression
    clf = LogisticRegression(max_iter=100000).fit(X, y)
    clf.predict(test_feats)
    print("Logistic Regression's Score on spam: " + str(clf.score(X, y)))


def predict_class():
    num_data = 887
    num_feats = 6
    num_training = int(4 * num_data / 5)
    read = pd.read_csv("titanic_multi_class.csv")
    data = np.array(read)
    np.random.shuffle(data)
    X = data[:num_training, :num_feats]
    y = data[:num_training, num_feats]
    test_feats = data[num_training:, :num_feats]
    test_labels = data[num_training:, num_feats]
    y = y.astype('int')
    test_labels = test_labels.astype('int')
    #######################################################################
    # Logistic Regression
    clf = LogisticRegression(max_iter=10000).fit(X, y)
    clf.predict(test_feats)
    print("Logistic Regression's Score on multi: " + str(clf.score(X, y)))


if __name__ == "__main__":
    predict_survival()
