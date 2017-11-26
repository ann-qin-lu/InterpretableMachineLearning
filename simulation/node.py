import numpy as np
class Node(object):

    def __init__(self, data_index, left_kid=None, right_kid=None):
        '''
        :param data_index: data_index in this node
        :param label: local label
        :param y: global label vector
        '''
        self.data_index = data_index
        self.left_kid = left_kid
        self.right_kid = right_kid

    def get_index(self):
        return self.data_index

    def set_kids(self, left_kid, right_kid):
        self.left_kid = left_kid
        self.right_kid = right_kid



if __name__ == "__main__":

    ls = []
    node1 = Node(np.array([1,2,3]))
    node2 = Node(np.array([2,3,4]))
    ls.append(node1)
    ls.append(node2)
    ls.remove(node1)
    print(len(ls))
    print(ls[0].get_index())
    node3 = Node(np.array([2,3,4]))
    ls.remove(node3)
    print(len(ls))


