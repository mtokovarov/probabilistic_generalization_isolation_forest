The repository contains the implementation of Probabilitic Generalization of Isolation Forest, presented in the article "A Probabilistic Generalization of Isolation Forest" by Tokovarov and Karczmarek.
It also contains the data that was used in numerical experiments. Please refer to the article for detailed description of the method and experiment procedure.

Contents:

artificial_data - artificially generated data applied in the experiment

real_data - real datasets applied in the experiment

isolation_forest - core class definitions

test_generalized_artificial_data.py - example of testing PGIF on artificially generated data

test_generalized_real_data.py - example of testing PGIF on artificially real data

utils.py - auxiliary functions: saving, loading

generating_datasets.py - generation of artificial datasets with outliers

