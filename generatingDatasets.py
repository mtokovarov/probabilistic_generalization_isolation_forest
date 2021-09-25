import numpy as np
import math

class Cluster:
    def __init__(self, dim_cnt, min_radius, max_radius, scale=1):
        self.center = np.random.rand(dim_cnt)
        self.radius = np.random.rand()*(max_radius - min_radius) + min_radius
    
    def get_center(self):
        return self.center
    
    def get_radius(self):
        return self.radius
        
    def if_intersects(self, cluster):
        if np.linalg.norm(self.center - cluster.get_center()) <= self.radius+cluster.get_radius():
            return True
        return False
    
    def in_cluster(self, points, closeness_tolerance = 0):
        ranges = np.linalg.norm(points - self.center, axis = 1)
        return ranges <= (self.radius+closeness_tolerance)
    
    def get_volume(self):
        dim_cnt = self.center.shape[0]
        k = dim_cnt // 2
        R = self.radius
        if (dim_cnt%2 == 0):
            volume = np.pi**k/math.factorial(k)*R**(2*k)
        else:
            volume = 2**(k+1)*np.pi**k/math.factorial(2*k+1)*R**(2*k+1)          
        return volume
        

class DatasetGenerator:
    def __init__(self, cluster_cnt, dim_cnt, total_sample_num, outlier_perc,
                 min_radius, max_radius, shuffle = False, 
                 closeness_tolerance = 0, scale = 1, overlapping = False):
        self.total_sample_num = total_sample_num
        self.dim_cnt = dim_cnt
        self.cluster_cnt = cluster_cnt
        self.outlier_perc = outlier_perc
        self.scale = scale
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.normal_cnt = int(total_sample_num*(1-self.outlier_perc/100))
        self.outlier_cnt = self.total_sample_num - self.normal_cnt
        self.shuffle  = shuffle
        self.overlapping = overlapping
        self.closeness_tolerance = closeness_tolerance
        
    def generate_data(self):
        self.define_clusters()
        self.normal_samples = self.make_samples(self.normal_cnt, 'normal')
        self.outlier_samples = self.make_samples(self.outlier_cnt, 'outlier')
        self.X = np.concatenate([self.normal_samples, self.outlier_samples], axis = 0)
        self.y = np.concatenate([np.zeros(self.normal_samples.shape[0]),
                           np.ones(self.outlier_samples.shape[0])])
        if (self.shuffle):
            inds = np.arange(self.X.shape[0])
            np.random.shuffle(inds)
            self.X = self.X[inds,...]
            self.y = self.y[inds,...]
            
            
    def define_clusters(self):
        self.clusters  = []
        for i in range(self.cluster_cnt):
            cluster = Cluster(self.dim_cnt, self.min_radius, self.max_radius)
            if(not self.overlapping):
                while(self.if_cluster_intersects_with_others(cluster)):
                    cluster = Cluster(self.dim_cnt, self.min_radius, self.max_radius)
            self.clusters.append(cluster)
    
    def make_samples(self, sample_cnt, sample_type = 'normal'):
        samples = np.zeros((sample_cnt,self.dim_cnt))
        portion_size = sample_cnt//10
        start = 0
        if (sample_type == 'normal'):
            aggreagting_operation = lambda x: np.sum(x.astype(np.uint8), axis = 0) > 0
            closeness_tolerance = 0
        if (sample_type == 'outlier'):
            aggreagting_operation = lambda x: np.sum(x.astype(np.uint8), axis = 0) == 0
            closeness_tolerance = self.closeness_tolerance
        while(start < sample_cnt):
            portion = np.random.rand(portion_size, self.dim_cnt)
            condition = aggreagting_operation(np.array([c.in_cluster(portion,
                                        closeness_tolerance) for c in self.clusters]))
            portion = portion[condition,...]
            end = start + portion.shape[0]
            if(end > sample_cnt):
                end = sample_cnt
            samples[start:end,...] = portion[:end - start,...]
            start = end
        return samples      
            
    def if_cluster_intersects_with_others(self, checked_cluster):
        for cluster in self.clusters:
            if (cluster.if_intersects(checked_cluster)):
                return True
        return False
    def get_data(self):
        return self.X, self.y
    
    def get_occupied_volume(self):
        point_cnt = int(1e7)
        points = np.random.rand(point_cnt, self.dim_cnt)
        in_clusters = np.sum(np.array([c.in_cluster(points, 0) for c in self.clusters]), axis = 0)>0
        field = np.sum(in_clusters)/point_cnt
        return field
        
    
from matplotlib import pyplot as plt
if __name__ == '__main__':
    cluster_cnt = 5
    dim_cnt = 2
    total_sample_num = 2000
    outlier_perc = 0.5
    min_radius = 0.1
    max_radius = 0.3
    shuffle = True
    closeness_tolerance = 0.05
    overlapping  = False
    dg = DatasetGenerator(cluster_cnt, dim_cnt, total_sample_num, outlier_perc,
                          min_radius, max_radius, shuffle, closeness_tolerance,
                          overlapping = overlapping)
    dg.generate_data()
    X, y = dg.get_data()
    plt.scatter(X[:,0], X[:,1], c = y, s = 5)
    
        