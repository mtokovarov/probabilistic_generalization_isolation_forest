import numpy as np
from node import Node
from tree import Tree


class TreeGrowerBasic:
    def __init__(self, X, sample_size):
        self.X = X
        self.dim_cnt = self.X.shape[1]
        self.sample_size = sample_size
        self.indeces = np.arange(0, self.X.shape[0])
        self.X_samples = []
    
    def make_train_datasets(self, ds_cnt):
        for i in range(ds_cnt):
            self.X_samples.append(self.make_sample_dataset())
    
    def make_empty_tree(self):
        return Tree()
    
    def grow_forest(self, tree_cnt, limit):
        self.make_train_datasets(tree_cnt)
        trees = [self.make_tree(X_sample, limit) for X_sample in self.X_samples]
        return trees
    
    def regrow_trees(self, limit = None):
        if (limit is None):
            limit = int(np.ceil(np.log2(self.sample_size)))
        trees = []
        for i in range(len(self.X_samples)):
            self.grown_tree = Tree()
            self.grown_tree.root = self.recursively_grow(self.X_samples[i], 0, limit) 
            trees.append(self.grown_tree)
        return trees        
    
    def make_tree(self, X_sample = None, limit = None):
        if (limit is None):
            limit = int(np.ceil(np.log2(self.sample_size)))
        if (X_sample is None):
            X_sample = self.make_sample_dataset()
            self.X_samples.append(X_sample)
        
        self.grown_tree = self.make_empty_tree()
        self.grown_tree.root = self.recursively_grow(X_sample, 0, limit) 
        return self.grown_tree
    
    def recursively_grow(self, X, tree_depth,depth_limit):
        self.grown_tree.tree_depth = tree_depth
        if tree_depth >= depth_limit or len(X) <= 1:
            self.grown_tree.exnodes += 1
            return Node(X, self.grown_tree.rot_op, self.grown_tree.border, tree_depth, 
                        left = None, right = None, node_type = 'exNode' )
        else:
            self.grown_tree.rot_op = self.get_rot_operator(X)
            X_rot = X[:,self.grown_tree.rot_op]
            if X_rot.min()==X_rot.max():
                self.grown_tree.exnodes += 1
                return Node(X, self.grown_tree.rot_op, self.grown_tree.border, 
                        tree_depth, left = None, right = None, node_type = 'exNode' )
            self.grown_tree.border = self.get_border(X_rot)
            w = np.where(X_rot <= self.grown_tree.border,True,False)
            return Node(X, self.grown_tree.rot_op, self.grown_tree.border, tree_depth,\
            left=self.recursively_grow(X[w,:], tree_depth+1, depth_limit),\
            right=self.recursively_grow(X[~w,:],tree_depth+1,depth_limit), node_type = 'inNode' )   
                    
    def make_sample_dataset(self):
        selected_indeces = np.random.choice(self.indeces, self.sample_size, replace = False)
        return self.X[selected_indeces,:]
    
    def getDataTransformer(self):
        return None
        
    
    
    #methods to be redefined in the child classes - we can modify the axis selection and split value generation
    
    def get_rot_operator(self, X):
        index = int(np.random.choice(np.arange(0, self.dim_cnt)))
        return index
           
    def get_border(self, X):
        min_val = min(X)
        max_val = max(X)
        return np.random.rand()*(max_val - min_val) + min_val
        