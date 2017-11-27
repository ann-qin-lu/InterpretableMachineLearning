from aggregate_sl.utilities import Utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from itertools import combinations
import rpy2
import matplotlib.pyplot as plt
from aggregate_sl.utilities import Utils
from sklearn.ensemble import RandomForestClassifier
import sys
import os
import pickle

def python_train_rf(X, y, model_name):

    model = os.path.join("../blackbox_models", model_name)
    if not os.path.isfile(model):
        clf = RandomForestClassifier()
        clf.fit(X,y)
        pickle.dump(clf, open(model, 'wb'))

def python_rf_wrapper(x, model_name):
    model = os.path.join("../blackbox_models", model_name)
    if not os.path.isfile(model):
        raise ValueError('No such model named {}'.format(model_name))
    clf = pickle.load(open(model, 'rb'))
    return clf.predict(x)


def plot_decision_bound(coefs, x, y):
    m = len(coefs)
    plt.plot(x[y==-1,0], x[y==-1,1], 'ro')
    plt.plot(x[y==1,0], x[y==1,1], 'g^')
    colors = ['b', 'y', 'm', 'k']
    for i in range(m):
        plt.plot(np.sort(x[:,0]), -coefs[i][0]*np.sort(x[:,0])/coefs[i][1], color=colors[i], ls='-')
    plt.show()


def test_sub_sampling(scale):
    point = [0, -0.5, 0.5]
    print(Utils.sub_sampling(point, 10, -1, 1, scale=scale))


def test_select_feature_lr_wrapper(mode):
    # general test data and label
    n, m = 500, 8
    coef = np.zeros((m,))
    coef[0] = 0.5
    coef[1] = -0.2
    x = np.random.uniform(-1, 1, size=(n, m))
    y = np.dot(x, coef)

    y[y>=0] = 1
    y[ y<0 ] = -1

    '''
    lr_refit = LogisticRegression(fit_intercept=False)
    lr_refit.fit(x[:, :2], y)


    y_hat = np.dot(x[:,:2], lr_refit.coef_[0])
    y_hat[y_hat>=0] = 1
    y_hat[y_hat<0] = -1

    print(np.mean(y_hat != y))

    print(lr_refit.coef_[0])
    print(lr_refit.intercept_)

    coefs=[np.array([0.5, -0.2]), lr_refit.coef_[0]]

    plot_decision_bound(coefs, x[:,:2], y)

    '''

    return Utils.select_feature_lr_wrapper(2, x, y, mode)



if __name__ == "__main__":
    test_sub_sampling(0.1)


