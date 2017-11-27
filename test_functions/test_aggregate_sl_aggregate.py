from aggregate_sl.utilities import Utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from itertools import combinations
import rpy2
from aggregate_sl.aggregate import AggregateLocalSL
from simulation.simulator import Simulator
from sklearn.ensemble import RandomForestClassifier
from aggregate_sl.utilities import Utils
import sys
import os
import pickle

#create a toy example in simulation

n_d = 200
n_f = 5
ls_split_features = [[0],[1, -1]]
ls_split_values = [[-0.2],[.5, None]]
leaf_feature_ls = [[2,3],[3,4],[1,4]]
leaf_beta_ls = [np.array([0, -0.5, 0.2]), np.array([0, 0.2, -0.4]), np.array([0, 0.1, 0.2])]
simulator = Simulator(n_d, n_f, ls_split_features, ls_split_values, leaf_feature_ls, leaf_beta_ls)
#simulator.ddt.plot_all_leafs()

def python_train_rf(X, y, model_name):

    model = os.path.join("../blackbox_models", model_name)
    if not os.path.isfile(model):
        clf = RandomForestClassifier()
        clf.fit(X,y)
        pickle.dump(clf, open(model, 'wb'))



def python_rf_wrapper(x):
    model = os.path.join("../blackbox_models", 'prf')
    if not os.path.isfile(model):
        raise ValueError('No such model named {}'.format('prf'))
    clf = pickle.load(open(model, 'rb'))
    return clf.predict(x)



X,y = simulator.get_simulated_data()
python_train_rf(X, y, 'prf')
y_hat = python_rf_wrapper(X)
accu = sum(y_hat == y)/float(len(y))
print("training accurancy is {}".format(accu))

data = X
k_sparse = 2
final_num_clusters = 3
aggregator = AggregateLocalSL(data, Utils.sub_sampling, python_rf_wrapper, k_sparse, final_num_clusters,
    fit_intercept=False, k_neighbor=1, sub_sampling_size=100, error_threshold=1)

aggregator.merge()
aggregator.summary(visulization = True)

