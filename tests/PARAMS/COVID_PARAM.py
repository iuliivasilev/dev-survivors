short = False

CRAID_param_grid = {
    "depth": [10],
    "criterion": ["peto"] if short else ["peto", "tarone-ware", "wilcoxon", "logrank"],
    "min_samples_leaf": [10] if short else [500, 1000, 2000],
    'cut': [False],
    "woe": [False],  # if short else [True, False],
    "signif": [0.05] if short else [0.05, 0.1, 0.15],
    "max_features": [1.0],
    "n_jobs": [40]
}

BSTR_param_grid = {
    "size_sample": [0.7], 
    "n_estimators": [10] if short else [10, 30], 
    "depth": [10],
    "ens_metric_name": ["conc", "ibs"],
    # "woe" : [], 
    "criterion": ["peto"] if short else ["peto", "tarone-ware", "wilcoxon", "logrank"], 
    "min_samples_leaf": [500, 1000, 2000],
    "max_features": [0.7] if short else ["sqrt"],
    "n_jobs": [40]
}

BOOST_param_grid = {
    "size_sample": [0.5] if short else [0.5],  # , 0.7],
    "n_estimators": [20],
    "ens_metric_name": ["ibs"] if short else ["roc", "ibs"],  # "conc",
    "depth": [10],
    "mode_wei": ['exp'] if short else ['square', "exp", "sigmoid", "softmax"],
    # "woe" : [],
    "criterion": ["logrank"] if short else ["confident", "confident_weights", "weights",
                                            "peto", "tarone-ware", "wilcoxon", "logrank"],
    "min_samples_leaf": [10] if short else [100],  # [500, 1000, 2000],
    "max_features": [0.3],  # ["sqrt"],
    "aggreg_func": ['wei'] if short else ['wei', 'mean'],
    "leaf_model": ["base"] if short else ["base_fast"],
    "all_weight": [False],  # , False],
    "n_jobs": [5],
    # ONLY SUM BOOSTING
    "learning_rate": [1.0, 0.2]
}

COVID_PARAMS = {
    "TREE": CRAID_param_grid,
    "BSTR": BSTR_param_grid,
    "BOOST": BOOST_param_grid,
    "SUMBOOST": BOOST_param_grid
}
