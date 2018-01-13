from simulation.node import Node
import numpy as np
import random
from sklearn.model_selection import train_test_split
import math
import numpy.random as rd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import sys
import time


class DeterministicDecisonTree(object):

    def __init__(self, X, Y, ls_split_features, ls_split_values, leaf_feature_ls, leaf_beta_ls):
        '''
        :param X: data, numpy array
        :param Y: labels, numpy array
        :param ls_split_featuress: the size of this list is the number of layers in the tree
        :param ls_split_values: value used to split
        :param leaf_feature_ls: features used for each leaf to fit a logistic regression
        :param leaf_beta_ls: betas used for each leaf to generate labels
        '''

        self.X = X
        self.Y = Y
        self.ls_split_features = ls_split_features
        self.ls_split_values = ls_split_values
        self.leaf_feature_ls = leaf_feature_ls
        self.leaf_beta_ls = leaf_beta_ls
        self.nodes = []
        self.leafs = []
        self.root = Node(np.array(range(len(Y))))
        self.nodes.append(self.root)
        self.leafs.append(self.root)
        self.finish_splitting = False

    def split(self, node, s_f, s_v, next_active_nodes):

        '''
        :param node: node to be splitted
        :param i_f: the index of feature to determine this split
        :param s_v: split value
        :param next_active_nodes: append two nodes after split
        :return:
        '''

        indexes = node.get_index()
        values = self.X[indexes, s_f]
        left_indexes = indexes[np.where(values <= s_v)]
        right_indexes = indexes[np.where(values > s_v)]
        left_node = Node(left_indexes)
        right_node = Node(right_indexes)
        node.set_kids(left_node, right_node)

        del_index = self.leafs.index(node)
        del self.leafs[del_index]
        self.leafs.insert(del_index, left_node)
        self.leafs.insert(del_index+1, right_node)
        self.nodes.append(left_node)
        self.nodes.append(right_node)
        next_active_nodes.append(left_node)
        next_active_nodes.append(right_node)

    def execute_splits(self):
        n_layers = len(self.ls_split_features)
        active_nodes = [self.root]
        for layer_iter in range(n_layers):
            next_active_nodes = []
            for a_i in range(len(active_nodes)):
                s_f = self.ls_split_features[layer_iter][a_i]
                if s_f == -1:
                    continue
                print("splitting index {}".format(a_i))
                leaf = self.leafs[a_i]
                s_v = self.ls_split_values[layer_iter][a_i]
                self.split(leaf, s_f, s_v, next_active_nodes)
            active_nodes = next_active_nodes
        self.finish_splitting = True

    def assign_labels(self):
        '''
        assign labels after finishing split
        :return: None
        '''

        if not self.finish_splitting:
            print("not finish splitting yet, please do splitting first by calling execute_splits")
            return
        for l_i in range(len(self.leafs)):
            leaf = self.leafs[l_i]
            ls_f = self.leaf_feature_ls[l_i]
            beta = self.leaf_beta_ls[l_i]
            indexes = leaf.get_index()
            values = self.X[indexes, :][:, ls_f]
            d_v = np.dot(values, beta[1:])+beta[0]
            self.Y[indexes[np.where(d_v >= 0)]] = 1
            self.Y[indexes[np.where(d_v < 0)]] = 0

    def get_data(self):
        return self.X, self.Y

    def refresh(self):
        self.nodes = []
        self.leafs = []
        self.root = Node(np.array(range(len(self.Y))))
        self.nodes.append(self.root)
        self.leafs.append(self.root)
        self.finish_splitting = False

    def print_tree_shape(self):

        def last_layer(ls):
            return not all(v is None for v in ls)

        layer = [self.root]
        while last_layer(layer):
            next_layer = []
            for node in layer:
                if not node:
                    print('-', end=' ')
                    next_layer.append(None)
                    next_layer.append(None)
                    continue
                print('*', end=' ')
                next_layer.append(node.left_kid)
                next_layer.append(node.right_kid)
            print('')
            layer = next_layer

    def plot_leaf(self, leaf_ind):
        node = self.leafs[leaf_ind]
        def close_event():
            plt.close()  # timer calls this function after 3 seconds and closes the window
        f_ls = self.leaf_feature_ls[leaf_ind]

        leaf_X = self.X[self.leafs[leaf_ind].get_index(),:][:,f_ls]
        x_1 = leaf_X[:,0]
        x_2 = leaf_X
        leaf_Y = self.Y[self.leafs[leaf_ind].get_index()]
        beta0, beta1, beta2 = self.leaf_beta_ls[leaf_ind]
        print(self.leaf_beta_ls[leaf_ind])

        plt.plot(leaf_X[leaf_Y == 0, 0], leaf_X[leaf_Y == 0, 1], 'ro')
        plt.plot(leaf_X[leaf_Y == 1, 0], leaf_X[leaf_Y == 1, 1], 'bs')
        plt.plot(x_1, -beta0/beta2-beta1/beta2*x_1, 'k-')
        plt.pause(2)
        plt.close()

    def plot_all_leafs(self):
        for l_i in range(len(self.leafs)):
            self.plot_leaf(l_i)



















