import numpy as np
import sys
from aggregate_sl.utilities import Utils
from aggregate_sl.node import Node
import sys
sys.path.append('../settings')
import settings.constant as constant
import matplotlib.pyplot as plt

class AggregateLocalSL(object):

    def _update_centroid(self, node):
        '''
        :param node: update centroid of this node,
        :return:
        '''
        node.update_centroid(np.mean(self.data[node.get_index(),:], axis=0))

    def _extract_coefs(self, centroid):
        '''
        :param centroid: extract coefs from a given centroid by running ksparse logistic on sampled data
        :return: coefs got
        '''
        x_s = self.sampling_function(centroid, self.sub_sampling_size)
        y_s = self.black_box_model(x_s)
        #print(y_s)
        n_0 = sum(y_s==0)
        n_1 = sum(y_s==1)

        if self.fit_intercept:
            coef_length = self.n_feature+1
        else:
            coef_length = self.n_feature

        if n_0 == self.sub_sampling_size:
            return [constant.ZEROFILLING,]*coef_length

        if n_1 == self.sub_sampling_size:
            return [constant.ONEFILLING,]*coef_length

        sub_coefs, err, min_ls = Utils.select_feature_lr_wrapper(self.k_sparse, x_s, y_s, 'bs', self.fit_intercept)


        if err > self.error_treshold:
            return [constant.LARGEERROR,]*coef_length

        if self.fit_intercept:
            coefs = np.zeros((self.n_feature + 1,))
            coefs[0] = sub_coefs[0]
            coefs[min_ls + 1] = sub_coefs[1:]
        else:
            coefs = np.zeros((self.n_feature,))
            coefs[min_ls] = sub_coefs
        coefs = coefs / np.linalg.norm(coefs)
        return coefs

    @staticmethod
    def _similarity_between_coefs(coef1, coef2):
        if coef1[0] == constant.LARGEERROR or coef2[0] == constant.LARGEERROR:
            return -1*constant.LARGEERROR
        return -1*np.linalg.norm(coef1-coef2)


    def _warm_up(self, sub_n=100):
        for node in self.clusters:
            self._update_centroid(node) #get initial centroid
            coefs = self._extract_coefs(node.get_centroid())
            node.update_coefs(coefs)

    def _extract_nearest_k(self, node):
        #print(node.get_index())
        neighbors = self.distance_map[node]
        res = neighbors[:min(len(neighbors), self.nearest_k)]
        return [x[0] for x in res]

    def _contains(self, key_tuple, obj):

        '''
        :param tuple:
        :param obj:
        :return:
        '''

        for item in key_tuple:
            if item == obj:
                return True
        return False

    def _update_active_pairs(self):
        self.active_pairs = set()
        for node in self.clusters:
            for neighbor in self._extract_nearest_k(node):
                self.active_pairs.add((node, neighbor))
        return self.active_pairs

    '''
    section for merge nodes
    '''

    def _remove_cluster(self, cluster):
        '''
        :param cluster: cluster(node) to be deleted
        :return:
        '''
        # cluster list
        self.clusters.remove(cluster)

        # similar pairs
        delete_set = set()
        for key_tuple in self.similarity_scores:
            if self._contains(key_tuple, cluster):
                delete_set.add(key_tuple)
        for key_tuple in delete_set:
            del self.similarity_scores[key_tuple]
        #distance map
        del self.distance_map[cluster]
        for nd in self.distance_map:
            self.distance_map[nd] = list(filter(lambda x: x[0] != cluster, self.distance_map[nd]))

        #active_pairs will be updated after adding the cluster got from merge

    def _add_cluster(self, cluster):
        '''
        :param cluster: after merging, new cluster comes
        :return:
        '''
        # cluster list
        self.clusters.append(cluster)

        # similar scores
        for node in self.clusters:
            pair = (node, cluster)
            dist = AggregateLocalSL._similarity_between_coefs(node.get_coefs(), cluster.get_coefs())
            self.similarity_scores[pair] = dist
            pair = (cluster, node)
            self.similarity_scores[pair] = dist

        # distance map
        for node in self.distance_map:
            self.distance_map[node].append((cluster, np.linalg.norm(cluster.get_centroid()-node.get_centroid())))
            self.distance_map[node] = sorted(self.distance_map[node], key=lambda x: x[1])

        self.distance_map[cluster] = []
        for node in self.clusters:
            if node == cluster:
                continue
            self.distance_map[cluster].append((node, np.linalg.norm(cluster.get_centroid()-node.get_centroid())))
        self.distance_map[node] = sorted(self.distance_map[node], key=lambda x: x[1])

    # active_pairs
        self._update_active_pairs()

    def _aggregate_two_nodes(self, nd1, nd2):
        merged_index = nd1.get_index()
        merged_index.extend(nd2.get_index())
        merged_node = Node(merged_index)
        self._update_centroid(merged_node)
        coefs = self._extract_coefs(merged_node.get_centroid())
        merged_node.update_coefs(coefs)
        self._remove_cluster(nd1)
        self._remove_cluster(nd2)
        self._add_cluster(merged_node)


    def _find_pair_to_merge(self):
        '''
        find the pair in active pairs with most similar coefs
        :return: None if all large errors, else the pair
        '''

        final_s_score = float('-inf')
        final_pair = None
        print(len(self.active_pairs))
        for pair in self.active_pairs:
            #print(pair)
            if self.similarity_scores[pair] == constant.LARGEERROR:
                print("yes")
                continue
            s_score = self._similarity_between_coefs(pair[0].get_coefs(), pair[1].get_coefs())
            if final_s_score < s_score:
                final_s_score = s_score
                #print(final_s_score)
                final_pair = pair

        if not final_pair:
            return None

        return final_pair


    def large_err_clusters(self):
        '''
        :return: assign those large errors to nearest
        '''
        pass

    def __init__(self, data, sampling_function, black_box_model, k_sparse, final_num_clusters,
                 fit_intercept = False,  k_neighbor=1, sub_sampling_size=100, error_threshold=1):

        '''
        :param list_objects: list of objects to be clustered, each node only contains index domain
        :param sampling_function: sampling function: given a point and number of sample points, return n
        :param black_box_model : black_box_model to return label for a given point
        :param k : search nearest k neighbors to aggregate
        '''

        self.similarity_scores = {} #similarity_scores
        self.distance_map = {} #key: node; value: sorted list of pairs: (node, distance from centroid)
        self.nearest_k = k_neighbor #every time we only aggromate k nearest neighbors which default value is 1
        self.sampling_function = sampling_function
        self.black_box_model = black_box_model
        self.k_sparse = k_sparse
        self.sub_sampling_size = sub_sampling_size
        self.fit_intercept = fit_intercept
        self.n_feature = data.shape[1]
        self.data = data
        self.error_treshold = error_threshold
        self.similarity_scores = dict()
        self.distance_map = dict()
        self.clusters = []
        for i in range(len(data)):
            self.clusters.append(Node([i]))

        self._warm_up()
        print("finish warm up")
        self.active_pairs = set()
        self.final_number_clusters = final_num_clusters

        for nd1 in self.clusters:
            for nd2 in self.clusters:
                if nd1 == nd2:
                    continue
                self.similarity_scores[(nd1, nd2)] \
                    = AggregateLocalSL._similarity_between_coefs(nd1.get_coefs(), nd2.get_coefs())

        for nd1 in self.clusters:
            self.distance_map[nd1] = []
            for nd2 in self.clusters:
                #print(nd1.get_index())
                #print(nd2.get_index())
                if nd1 == nd2:
                    continue
                #print(nd1.get_centroid())
                self.distance_map[nd1]\
                    .append((nd2, np.linalg.norm(nd1.get_centroid()-nd2.get_centroid())))
            self.distance_map[nd1] = sorted(self.distance_map[nd1], key=lambda x: x[1])

        # print(self.distance_map[nd1])

        self._update_active_pairs()

    def merge(self):
        '''
        version which does not handle the large error centroid
        :return:
        '''
        while len(self.clusters)>self.final_number_clusters:
            pair = self._find_pair_to_merge()
            self._aggregate_two_nodes(pair[0], pair[1])


    def summary(self, visulization=False):

        '''
        :param visulization: visulization only could be set to be true only when
        active parameters are two
        :return:
        '''

        print("there are {} clusters".format(self.final_number_clusters))
        print("==============")
        for i in range(self.final_number_clusters):
            cluster = self.clusters[i]
            coefs = cluster.get_coefs()
            print(coefs)
            x_s = self.data[cluster.get_index(),:]
            y_s = self.black_box_model(x_s)
            y_hat = np.zeros(y_s.shape)
            if not self.fit_intercept:
                y_hat[np.dot(x_s, coefs)>0] = 1
            else:
                y_hat[np.dot(x_s, coefs[1:])+coefs[0]>0]=1
            accuracy = sum(y_hat == y_s)/float(len(y_s))
            if self.fit_intercept:
                active_ls = np.where(abs(coefs[1:])>constant.MINIMALNONZERO)[0]
            else:
                active_ls = np.where(abs(coefs)>constant.MINIMALNONZERO)[0]

            print("for cluster {}: importance feature {} and accuracy is {}".format(i, active_ls, accuracy))

        if not visulization:
            return

        for i in range(self.final_number_clusters):
            cluster = self.clusters[i]
            coefs = cluster.get_coefs()
            x_s = self.data[cluster.get_index(),:]
            print(x_s.shape)
            y_s = self.black_box_model(x_s)
            x_active = x_s[:,active_ls]
            print(x_active.shape)
            x_1 = x_active[:,0]
            x_2 = x_active[:,1]

            beta0 = int(self.fit_intercept)
            if self.fit_intercept:
                beta1 = coefs[1]
                beta2 = coefs[2]
            else:
                beta1 = coefs[0]
                beta2 = coefs[1]

            plt.plot(x_1[y_s == 0,], 'ro')
            plt.plot(x_2[y_s == 1,], 'bs')
            plt.plot(x_1, -beta0 / beta2 - beta1 / beta2 * x_1, 'k-')
            plt.pause(3)
            plt.close()



if __name__ == "__main__":
    print(constant.ZEROFILLING )

