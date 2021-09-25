import numpy as np

class Tree(object):
    def __init__(self):
        self.tree_depth = 0
        self.border = None
        self.rot_op = None
        self.exnodes = 0
        self.root = None
        
    def c_factor(self, n) :
        return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))
    
        
    
    def find_path(self, x):
        
        def recourcievelyFindPath(T):  
            nonlocal e
            if T.ntype == 'exNode':
                if T.size > 1:
                    e = e + self.c_factor(T.size)
                return e
            else:
                e += 1
                if T.get_data_transformed_by_node(x) <= T.border:
                    return recourcievelyFindPath(T.left)
                else:
                    return recourcievelyFindPath(T.right)
        e = 0     
        recourcievelyFindPath(self.root)
        return e
        
        