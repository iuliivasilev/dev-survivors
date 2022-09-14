short = False

CRAID_param_grid = {
    "depth": [15],
    "criterion": ["peto"] if short else ["peto", "tarone-ware", "wilcoxon", "logrank"],
    "min_samples_leaf": [10] if short else [10, 15, 20],
    'cut': [True, False],
    "woe": [False],  # if short else [True, False],
    "signif": [0.05] if short else [0.05, 0.1, 0.15],
    "max_features": [1.0]
}

BSTR_param_grid = {
    "size_sample": [0.7], 
    "n_estimators": [10] if short else [10, 30], 
    "depth": [15],
    "ens_metric_name": ["conc", "ibs"],
    # "woe" : [], 
    "criterion": ["peto"] if short else ["peto", "tarone-ware", "wilcoxon", "logrank"], 
    "min_samples_leaf": [15, 20],
    "max_features": [0.7] if short else [0.5, 0.7]  # ["sqrt"]
}

BOOST_param_grid = {
    "size_sample": [0.5] if short else [0.5, 0.7],
    "n_estimators": [15] if short else [15], 
    "ens_metric_name": ["ibs"] if short else ["conc", "ibs"],
    "depth": [15],
    "mode_wei": ['exp'] if short else ['square', 'exp'],
    # "woe" : [],
    "criterion": ["logrank"] if short else ["peto", "tarone-ware", "wilcoxon", "logrank"],
    "min_samples_leaf": [25] if short else [15, 20, 25],
    "max_features": [0.7] if short else [0.7],
    "aggreg_func": ['wei'] if short else ['wei', 'mean']
}

GBSG_PARAMS = {
    "TREE": CRAID_param_grid,
    "BSTR": BSTR_param_grid,
    "BOOST": BOOST_param_grid
}
