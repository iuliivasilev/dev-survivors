short = False

CRAID_param_grid = {
    "depth": [10, 15],
    "balance": [None, "balance", "balance+correct", "balance+weights"],
    "criterion": ["peto"] if short else ["confident_weights"],  # ["weights"],  # ["peto", "tarone-ware", "wilcoxon", "logrank"],
    "min_samples_leaf": [10] if short else [30, 50],
    'cut': [False],
    "woe": [False],  # if short else [True, False],
    "signif": [0.05, 0.15],  # if short else [0.05, 0.1, 0.15],
    "max_features": [1.0],
    "n_jobs": [40]
}

BSTR_param_grid = {
    "size_sample": [0.7],
    "n_estimators": [10] if short else [10, 30], 
    "depth": [15, 20],
    "ens_metric_name": ["conc", "ibs", "roc"],
    # "woe" : [], 
    "criterion": ["peto"] if short else ["peto", "tarone-ware", "wilcoxon", "logrank"], 
    "min_samples_leaf": [30, 50],
    "max_features": [0.2] if short else [0.2, "sqrt"],  # ["sqrt"],
    "n_jobs": [40]
}

BOOST_param_grid = {
    "size_sample": [0.9],  # [0.5, 0.7],
    "n_estimators": [100],  # old [30]
    "ens_metric_name": ["ibs"] if short else ["bic", "roc", "ibs"],  # ["roc", "ibs"],  # ["conc", "ibs"]
    "depth": [10] if short else [5, 10],  # new [5]
    "mode_wei": ['exp'] if short else ['exp', 'square', "sigmoid", "softmax"],
    "criterion": ["confident", "logrank"] if short else ["confident", "confident_weights", "weights",
                                            "peto", "tarone-ware", "wilcoxon", "logrank"],
    "min_samples_leaf": [5] if short else [5, 100],  # [20, 100]
    "max_features": [0.4] if short else [0.5, 0.7],  # [0.2, 0.5],
    "aggreg_func": ['wei'] if short else ['wei', 'mean'],
    "leaf_model": ["base_fast"] if short else ["base_fast"],
    "all_weight": [False],
    "balance": [None, "balance", "balance+weights"],
    "with_arc": [True, False],
    "n_jobs": [2]
}

SUMBOOST_param_grid = BOOST_param_grid.copy()
SUMBOOST_param_grid["learning_rate"] = [1.0, 0.2]

PROBOOST_param_grid = BOOST_param_grid.copy()
del PROBOOST_param_grid["mode_wei"]

ONK_PARAMS = {
    "TREE": CRAID_param_grid,
    "BSTR": BSTR_param_grid,
    "BOOST": BOOST_param_grid,
    "SUMBOOST": SUMBOOST_param_grid,
    "PROBOOST": PROBOOST_param_grid
}
