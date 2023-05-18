short = False

CRAID_param_grid = {
    "depth": [10],
    "balance": [None],  # [None, "balance", "balance+correct", "only_log_rank"]
    "criterion": ["peto"] if short else ["confident", "peto", "tarone-ware", "wilcoxon", "logrank"],
    "min_samples_leaf": [10] if short else [100, 500, 1000],
    "leaf_model": ["base_fast"] if short else ["base_fast"],
    'cut': [False, True],
    "woe": [True],  # if short else [True, False],
    "signif": [0.05] if short else [0.05, 1.0],
    "max_features": [1.0],
    "n_jobs": [10]
}

BSTR_param_grid = {
    "size_sample": [0.5],
    "n_estimators": [10] if short else [30],
    "depth": [10],
    "ens_metric_name": ["bic", "roc", "ibs"],
    "criterion": ["peto"] if short else ["confident",
                                         "peto", "tarone-ware", "wilcoxon", "logrank"],
    "leaf_model": ["base_fast"] if short else ["base_fast"],
    "balance": [None],  # [None, "balance", "balance+correct", "only_log_rank"]
    "min_samples_leaf": [100],
    "max_features": [0.7] if short else [0.3],
    "n_jobs": [10]
}

BOOST_param_grid = {
    "size_sample": [0.5] if short else [0.5],  # , 0.7],
    "n_estimators": [30],  # new 50
    "ens_metric_name": ["ibs"] if short else ["bic", "roc", "ibs"],  # ["roc", "ibs"],  # "conc",
    "depth": [10],  # new 5
    "mode_wei": ['exp'] if short else ['square', "exp", "sigmoid", "softmax"],
    "criterion": ["confident", "logrank"] if short else ["confident", "weights",
                                            "peto", "tarone-ware", "wilcoxon", "logrank"],
    "min_samples_leaf": [10] if short else [100],  # new 1000
    "max_features": [0.3],  # ["sqrt"],
    "aggreg_func": ['wei'] if short else ['wei', 'mean'],
    "leaf_model": ["base"] if short else ["base_fast"],
    "all_weight": [False],  # , False],
    "balance": [None, "balance", "only_log_rank"],
    "with_arc": [False],
    "n_jobs": [5]
}

SUMBOOST_param_grid = BOOST_param_grid.copy()
SUMBOOST_param_grid["learning_rate"] = [1.0, 0.2]

PROBOOST_param_grid = BOOST_param_grid.copy()
del PROBOOST_param_grid["mode_wei"]

COVID_PARAMS = {
    "TREE": CRAID_param_grid,
    "BSTR": BSTR_param_grid,
    "BOOST": BOOST_param_grid,
    "SUMBOOST": SUMBOOST_param_grid,
    "PROBOOST": PROBOOST_param_grid,
    "IBSBOOST": PROBOOST_param_grid,
    "IBSPROBOOST": PROBOOST_param_grid
}
