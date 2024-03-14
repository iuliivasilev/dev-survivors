short = False

CRAID_param_grid = {
    "depth": [15],
    "balance": [None],  # [None, "balance", "balance+correct", "only_log_rank"]
    "criterion": ["peto"] if short else ["confident", "peto", "tarone-ware", "wilcoxon", "logrank"],
    "min_samples_leaf": [10] if short else [5, 15, 20],
    "leaf_model": ["base_fast"] if short else ["base_fast"],
    'cut': [True, False],
    "woe": [True],  # if short else [True, False],
    "signif": [0.05] if short else [0.05, 1.0],
    "leaf_model": ["base_fast"],
    "max_features": [1.0]
}

BSTR_param_grid = {
    "size_sample": [0.5, 0.7],  # 0.5
    "n_estimators": [10] if short else [30],
    "depth": [10],
    "ens_metric_name": ["IBS_REMAIN"],  # ["bic", "CI", "ibs"],
    "criterion": ["peto"] if short else ["confident", "peto", "tarone-ware", "wilcoxon", "logrank"],
    "leaf_model": ["base_fast"] if short else ["base_fast"],
    "balance": [None],  # [None, "balance", "balance+weights", "only_log_rank"]
    "min_samples_leaf": [5, 20],
    "max_features": [0.7] if short else ["sqrt", 0.3, 0.5]  # ["sqrt"]
}

BOOST_param_grid = {
    "size_sample": [0.5] if short else [0.5, 0.7],  # [0.5, 0.7]
    "n_estimators": [15] if short else [50],  # new 50
    "ens_metric_name": ["ibs"] if short else ["bic", "CI", "ibs"],  # ["roc", "CI", "ibs"],
    "depth": [10],  # new 5
    "mode_wei": ['exp'] if short else ['linear', 'square', "exp", "sigmoid", "softmax"],  # 'square', 'exp'],
    "criterion": ["confident", 'logrank'] if short else ["confident", "weights",
                                            "logrank", "peto", "tarone-ware", "wilcoxon"],
    "min_samples_leaf": [25] if short else [5, 20],  # new [20]
    "max_features": [0.7] if short else ["sqrt", 0.3, 0.5],  # "sqrt"
    "aggreg_func": ['wei'] if short else ['wei'],  # 'mean'
    "leaf_model": ["base"] if short else ["base_fast", "wei_survive"],  # "only_hazard", "base"],
    "all_weight": [False, True],
    "balance": [None],  # [None, "balance", "only_log_rank"],
    "with_arc": [False],
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
    "PROBOOST": PROBOOST_param_grid,
    "IBSBOOST": PROBOOST_param_grid,
    "IBSPROBOOST": PROBOOST_param_grid
}
