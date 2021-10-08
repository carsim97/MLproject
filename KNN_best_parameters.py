from secml.data import CDataset
from secml.ml import CClassifierKNN

from sklearn.metrics import roc_curve, det_curve, RocCurveDisplay, DetCurveDisplay

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# This function returns the ROC / DET curve value corresponding to a selected False Positive Rate.
def find_curve_value(curve_x, curve_y, fpr_ref=0.01, curve_type='roc'):
    if fpr_ref in curve_x:
        idx_ref = np.where(curve_x == fpr_ref)
        return curve_y[idx_ref]
    else:
        idx_less_than_ref = curve_x < fpr_ref
        idx_more_than_ref = curve_x > fpr_ref
        if len(curve_x[idx_less_than_ref]) and len(curve_x[idx_more_than_ref]):
            if curve_type == 'roc':
                x1 = (curve_x[idx_less_than_ref])[-1]
                y1 = curve_y[curve_x == x1][-1]
                x2 = (curve_x[idx_more_than_ref])[0]
                y2 = curve_y[curve_x == x2][0]
            elif curve_type == 'det':
                x1 = (curve_x[idx_less_than_ref])[0]
                y1 = curve_y[curve_x == x1][0]
                x2 = (curve_x[idx_more_than_ref])[-1]
                y2 = curve_y[curve_x == x2][-1]
            else:
                raise ValueError('Invalid curve type')
            m = (y2 - y1)/(x2 - x1)
            q = (x2 * y1 - x1 * y2)/(x2 - x1)
            y = m * fpr_ref + q
            return y
        else:
            return None


if __name__ == '__main__':

    # Loading training and test sets in CDataset format
    training = CDataset.load('ML Dataset/training_set.gz')
    test = CDataset.load('ML Dataset/test_set.gz')

    # Using K-Nearest Neighbors classifier with several values of parameter k
    classifiers = {
        "K-Nearest Neighbors (K=1)": CClassifierKNN(n_neighbors=1),
        "K-Nearest Neighbors (K=3)": CClassifierKNN(n_neighbors=3),
        "K-Nearest Neighbors (K=5)": CClassifierKNN(n_neighbors=5),
        "K-Nearest Neighbors (K=7)": CClassifierKNN(n_neighbors=7),
        "K-Nearest Neighbors (K=9)": CClassifierKNN(n_neighbors=9),
        "K-Nearest Neighbors (K=11)": CClassifierKNN(n_neighbors=11),
        "K-Nearest Neighbors (K=13)": CClassifierKNN(n_neighbors=13),
        "K-Nearest Neighbors (K=15)": CClassifierKNN(n_neighbors=15),
        "K-Nearest Neighbors (K=17)": CClassifierKNN(n_neighbors=17),
        "K-Nearest Neighbors (K=19)": CClassifierKNN(n_neighbors=19)
    }

    n_classifiers = len(classifiers.items())
    tpr_eval = np.zeros((n_classifiers, ))
    fnr_eval = np.zeros((n_classifiers, ))

    fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5))

    i = 0
    for name, classifier in classifiers.items():
        # Training every classifier
        classifier.fit(training.X, training.Y)

        # Computing decision function
        y_score = classifier.decision_function(test.X, y=1)

        y_true = test.Y.get_data()
        y_score = y_score.get_data()

        # Plotting ROC curve
        fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
        roc_fig = RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=name, pos_label=1)
        roc_fig.plot(ax=ax_roc, name=name)
        tpr_eval[i] = find_curve_value(fpr, tpr, curve_type='roc')

        # Plotting DET curve
        fpr, fnr, _ = det_curve(y_true=y_true, y_score=y_score)
        det_fig = DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name=name, pos_label=1)
        det_fig.plot(ax=ax_det, name=name)
        fnr_eval[i] = find_curve_value(fpr, fnr, curve_type='det')

        print(name)
        print('True Positive Rate (at FPR = 1%):', tpr_eval[i])
        print('False Negative Rate (at FPR = 1%):', fnr_eval[i])
        print()

        i = i + 1

    ref = 0.01
    ax_roc.axvline(x=ref, linestyle='--', color='k')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) curves')

    ref = sp.stats.norm.ppf(ref)
    ax_det.axvline(x=ref, linestyle='--', color='k')
    ax_det.set_title('Detection Error Tradeoff (DET) curves')

    plt.legend()

    fig1, [ax_tpr, ax_fnr] = plt.subplots(1, 2, figsize=(11, 5))

    # Plotting TPR and FNR at false positive rate = 1.0%, as a function of K
    K = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    ax_tpr.plot(K, tpr_eval)
    ax_tpr.set_title('K-Nearest Neighbors: TPR (at FPR = 1%)')
    ax_tpr.set_xlabel('K')
    ax_tpr.set_ylabel('True Positive Rate (at FPR = 1%)')
    ax_tpr.set_xticks(K)
    ax_fnr.plot(K, fnr_eval)
    ax_fnr.set_title('K-Nearest Neighbors: FNR (at FPR = 1%)')
    ax_fnr.set_xlabel('K')
    ax_fnr.set_ylabel('False Negative Rate (at FPR = 1%)')
    ax_fnr.set_xticks(K)

    plt.show()
