import numpy as np
import os
from simulation.decision_tree import DeterministicDecisonTree
import pickle


class Simulator(object):

    def __init__(self, n_d, n_f, ls_split_features, ls_split_values, leaf_feature_ls, leaf_beta_ls):
        '''
        :param n_d: number of data generated
        :param n_f: number of features generated
        :param list_split_features: list of features used to split data
        :param leaf_feature_index_list: leafs for each leafs
        :param leaf_beta_list: beta for logistic regression in each leaf
        '''

        #uniform generate x of size n_d x n_f

        self.X = np.random.uniform(-1, 1, size=(n_d, n_f))
        self.Y = np.zeros((len(self.X),))
        self.ddt = DeterministicDecisonTree(self.X, self.Y, ls_split_features, ls_split_values, leaf_feature_ls, leaf_beta_ls)
        self.ddt.execute_splits()
        self.ddt.assign_labels()
        self.ddt.print_tree_shape()

    def print_split_information(self, output_file):
        '''
        print split information and output it to a file
        :return:
        '''
        print("features used to split")
        for i in range(self.ls_split_features):
            print("layer {}:".format(i))
            print(self.ls_split_features[i])

        print("leafs logistic features")
        for i in range(self.leaf_feature_ls):
            print("leaf {}:".format(i))
            print(self.leaf_feature_ls[i])

        print("beta used for leaf")
        for i in range(self.leaf_beta_ls):
            print("leaf {}:".format(i))
            print(self.leaf_beta_ls[i])

        with open(output_file, 'w') as f:  # Python 3: open(..., 'wb')
            pickle.dump([self.ls_split_feature, self.leaf_feature_ls, self.leaf_beta_ls, self.X, self.Y], f)

    def get_simulated_data(self, output_file = None):

        if not output_file:
            with open(output_file, 'w') as f:  # Python 3: open(..., 'wb')
                pickle.dump([self.X, self.Y], f)

        return self.X, self.Y


if __name__ == "__main__":

    n_d = 2000
    n_f = 6
    ls_split_features = [[1],[2, -1],[-1, 4]]
    ls_split_values = [[0.2],[.5, None],[None, 0.7]]
    leaf_feature_ls = [[1,4],[3,5],[2,3],[1,2]]
    leaf_beta_ls = [np.array([0, -0.5, 0.2]), np.array([0, 0.2, 0.4]), np.array([0, 0.1, 0.2]), np.array([0,0.3,-0.2])]
    simulator = Simulator(n_d, n_f, ls_split_features, ls_split_values, leaf_feature_ls, leaf_beta_ls)
    simulator.ddt.plot_all_leafs()























