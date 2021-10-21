import numpy as np
import pickle
import pandas as pd
from scipy.io import loadmat

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def load_data(sourcePath, dataset):
    
    if 'mat' in dataset:
        mat = loadmat(f"{sourcePath}/{dataset}")
        X = mat['X']
        y = np.squeeze(mat['y'])
        mins = X.min(axis = 0)
        maxes = X.max(axis = 0)
        X = (X - mins)/(maxes-mins + np.spacing(maxes-mins))
        return X,y
    
    if ('csv' in dataset):
        data = pd.read_csv(f"{sourcePath}/{dataset}", "rb", delimiter=",")
        data = data.values
        
    else:
        data = np.loadtxt(f"{sourcePath}/{dataset}", delimiter = ',')
    
    dataset = "_".join(dataset.split(".")[:-1])
    X = data[:,:-1]
    X = X.astype(np.float)
    mins = X.min(axis = 0)
    maxes = X.max(axis = 0)
    X = (X - mins)/(maxes-mins + np.spacing(maxes-mins))
    y = data[:,-1]
    if 'mulcross' in dataset:
        y = (y != "'Normal'").astype(np.uint8)
        
    return X, y# -*- coding: utf-8 -*-

