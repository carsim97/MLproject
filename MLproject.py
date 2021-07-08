from secml.data import CDataset
from secml.ml import CClassifierSVM
from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    training = CDataset.load('ML Dataset/training_set.gz')
    test = CDataset.load('ML Dataset/test_set.gz')

    classifier1 = SVC()
    x_training_tot = training.X.get_data()
    y_training_tot = training.Y.get_data()
    x_test = test.X.get_data()
    y_test = test.Y.get_data()

    x_training = x_training_tot
    y_training = y_training_tot
    classifier1.fit(x_training, y_training)

    y_pred = classifier1.predict(x_test)

    accuracy = (y_test == y_pred).mean() * 100

    print('Accuracy (SKLEARN):', accuracy)

    classifier2 = CClassifierSVM()
    x_training_tot = training.X
    y_training_tot = training.Y
    x_test = test.X
    y_test = test.Y

    x_training = x_training_tot
    y_training = y_training_tot
    classifier2.fit(x_training, y_training)

    y_pred = classifier2.predict(x_test)

    accuracy = (y_test == y_pred).mean() * 100

    print('Accuracy (SECML):', accuracy)

