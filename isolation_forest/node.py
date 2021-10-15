import numpy as np

class Node(object):
    def __init__(self, X, rot_op, border, e, left, right, node_type = ''):
        self.e = e
        self.size = len(X)
        self.X = X
        self.rot_op = rot_op
        self.border = border  #border
        self.left = left
        self.right = right
        self.ntype = node_type
    
    def get_data_transformed_by_node(self, data):
        return data[self.rot_op]