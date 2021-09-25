from tree_grower_basic import TreeGrowerBasic
import numpy as np

class TreeGrowerGeneralized(TreeGrowerBasic): #abstract class of generalized Isolation Grower
    def __init__(self, X, sample_size, power):
        super(TreeGrowerGeneralized, self).__init__(X, sample_size)
        self.power = power
        
    def integratedProbabilityDensity(self, x):
        raise Exception("The class is abstract, instanciate its child!")
        
    def probabilityDensity(self, x):
        raise Exception("The class is abstract, instanciate its child!")
    
    def generateRandomNumber_01(self):
        p = np.random.rand()
        return p
 
    def cumulatedProbabilityDensity(self, x):
        #we consider only bounded support kernel functions with support [-1, 1]
        #for any function of that kind the value of Kintegrated(-1) == 1/2
        return self.integratedProbabilityDensity(x) + 0.5 
      
    def newtonRaphson(self, target, eps, func, func_derivative):
        #we always start at x = 0, hence the cum probablity is equal to 0.5
        x = 0
        y = 0.5 - target
        while(abs(y) > eps):
            x -= y/func_derivative(x)            
            if x < -1:
                x = -np.random.rand()
            elif x > 1:
                x = np.random.rand()
            y = func(x) - target
                
        return x
    
        
    def invertedCumulatedProbabilityFunc(self, a, b, pStar):
        #the most reasonable solution for xstart is 0, as the prob density (hense, derivative of analyzed function)
        #is the highest in 0 for the majority of kernels
        u = self.newtonRaphson(pStar, 1e-6, self.cumulatedProbabilityDensity, self.probabilityDensity)
        x = (b-a)/2*u+(b+a)/2
        return x   
            
    def prepare_X(self, X):
        X_sorted = np.unique(X)
        return X_sorted
        
    def get_border(self, X):
        X_sorted = self.prepare_X(X)
        dists = np.diff(X_sorted)
        cumulated_probs = dists**(self.power+1)/2
        m = np.sum(cumulated_probs)
        cumulated_probs /= m
        i=0
        p = np.random.rand()#
        while(cumulated_probs[i]<p):
            p -= cumulated_probs[i]
            i+=1
        pStar = p/cumulated_probs[i]
        return self.invertedCumulatedProbabilityFunc(X_sorted[i], X_sorted[i+1], pStar)