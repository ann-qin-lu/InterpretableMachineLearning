import numpy as np
class Node(object):

    def __init__(self, index, centroid = None, coefs = None):
        self.index = index
        self.coefs = coefs
        self.centroid = centroid

    def update_index(self, x):
        self.index = x

    def update_centroid(self, centroid):
        if type(centroid) == list:
            centroid = np.array(centroid)
        self.centroid = centroid

    def update_coefs(self, coefs):
        if type(coefs) == list:
            coefs = np.array(coefs)
        self.coefs = coefs

    def get_index(self):
        return self.index

    def get_centroid(self):
        if type(self.centroid) == list:
            self.centroid = np.array(self.centroid)
        return self.centroid

    def get_coefs(self):
        return self.coefs

if __name__ == "__main__":


    v_map = {}
    nd1 = Node(1)
    nd2 = Node(2)
    nd3 = Node(3)
    v_map[nd1]=[]
    v_map[nd1].append(3)
    print(v_map[nd1])

    ls = [1,2,3,4,10,9]
    ls.remove(10)
    print(ls)