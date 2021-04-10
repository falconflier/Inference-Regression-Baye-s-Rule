import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from housing import stand_n, estimate


if __name__ == "__main__":
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
    print("Number of mislabeled points out of a total %d points : %d"% (test_feats.shape[0], (test_labels != y_pred).sum()))
    print("Naive Baye's Score: " + str(1 - (test_labels != y_pred).sum()/(test_feats.shape[0])))

    #######################################################################
    # Logistic Regression
    clf = LogisticRegression().fit(X, y)
    clf.predict(test_feats)
    print("Logistic Regression's Score: " + str(clf.score(X, y)))
