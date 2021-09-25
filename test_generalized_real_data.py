import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from isolation_forest.isolation_forest import IsolationForest
from isolation_forest.tree_grower_generalized_uniform import TreeGrowerGeneralizedUniform
from isolation_forest.tree_grower_basic import TreeGrowerBasic
import os
from utils import load_data

def test_grower_saving_files(grower, grower_args, source_path, target_path,
                repeat_cnt, additional_info=""):
    target_folder_path = f"{target_path}\\{grower.__name__}_{additional_info}"
    if(not os.path.exists(target_folder_path)):
        os.mkdir(target_folder_path)
    datasets = os.listdir(source_path)
    for dataset in datasets:
        ds_name = dataset.split('.')[0]
        X,y = load_data(source_path, dataset)
        total_sample_cnt = X.shape[0]
        scores = np.zeros((repeat_cnt, total_sample_cnt))
        for i in range(repeat_cnt):
            gr_args = (X,)+grower_args
            new_grower = grower(*gr_args)
            forest = IsolationForest(new_grower, X, tree_cnt, sample_size)
            forest.grow_forest()
            print(f"trained, {grower.__name__}!")
            scores[i,...] = forest.compute_paths()
            print(f'{ds_name}: {roc_auc_score(y, scores[i,...])}')
        np.save(f'{target_path}\\{dataset.split(".")[0]}.npy', scores)


tree_cnt = 100
sample_size = 256
repeat_cnt = 30
power = 2

growers = [TreeGrowerBasic, TreeGrowerGeneralizedUniform]
grower_arg_sets = [(sample_size,), (sample_size,power)]
additional_infos = ['', f'power_{power}']

source_path = 'real_data'
target_path = 'results_real'
if(not os.path.exists(target_path)):
    os.mkdir(target_path)

repeat_cnt = 1
cnt = 0
for grower, grower_arg_set, additional_info in zip(growers, grower_arg_sets, additional_infos):
    test_grower_saving_files(grower, grower_arg_set, source_path, target_path, 
                             repeat_cnt, additional_info)


