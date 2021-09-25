from tree_grower_generalized import TreeGrowerGeneralized
import numpy as np

class TreeGrowerGeneralizedUniform(TreeGrowerGeneralized):
    def integratedProbabilityDensity(self, x):
        return 0.5*x
        
    def probabilityDensity(self, x):
        return 0.5
