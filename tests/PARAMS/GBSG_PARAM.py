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
    "size_sample": [0.5] if short else [0.5],  # [0.5, 0.7]
    "n_estimators": [15] if short else [30],  # new 50
    "ens_metric_name": ["ibs"] if short else ["bic", "roc", "conc", "ibs"],  # ["roc", "conc", "ibs"],
    "depth": [10],  # new 5
    "mode_wei": ['exp'] if short else ['linear', 'square', "exp", "sigmoid", "softmax"],  # 'square', 'exp'],
    "criterion": ['logrank'] if short else ["confident", "confident_weights", "weights",
                                            "logrank", "peto", "tarone-ware", "wilcoxon"],
    "min_samples_leaf": [25] if short else [5, 20],  # new [20]
    "max_features": [0.7] if short else ["sqrt", 0.3, 0.5],  # "sqrt"
    "aggreg_func": ['wei'] if short else ['wei', 'mean'],
    "leaf_model": ["base"] if short else ["base_fast", "wei_survive"],  # "only_hazard", "base"],
    "all_weight": [False],
    "n_jobs": [2]
}

SUMBOOST_param_grid = BOOST_param_grid.copy()
SUMBOOST_param_grid["learning_rate"] = [1.0, 0.2]

PROBOOST_param_grid = BOOST_param_grid.copy()
del PROBOOST_param_grid["mode_wei"]

GBSG_PARAMS = {
    "TREE": CRAID_param_grid,
    "BSTR": BSTR_param_grid,
    "BOOST": BOOST_param_grid,
    "SUMBOOST": SUMBOOST_param_grid,
    "PROBOOST": PROBOOST_param_grid
}
