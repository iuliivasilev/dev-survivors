short = False

CRAID_param_grid = {
    "depth": [10],
    "balance": [None],  # [None, "balance", "balance+correct", "only_log_rank"]
    "criterion": ["peto"] if short else ["confident", "peto", "tarone-ware", "wilcoxon", "logrank"],
    "min_samples_leaf": [5] if short else [10, 25],
    "leaf_model": ["base_fast"] if short else ["base_fast"],
    'cut': [True, False],
    "woe": [True],  # if short else [True, False],
    "signif": [0.05] if short else [0.05, 1.0],
    "max_features": [1.0],
    "n_jobs": [10]
}

BSTR_param_grid = {
    "size_sample": [0.5, 0.7],  # 0.7
    "n_estimators": [10] if short else [20],
    "depth": [5],
    "ens_metric_name": ["IBS_REMAIN"],  # ["bic", "roc", "ibs"],
    "criterion": ["peto"] if short else ["confident", "peto", "tarone-ware", "wilcoxon", "logrank"],
    "leaf_model": ["base_fast"] if short else ["base_fast"],
    "balance": [None],  # [None, "balance", "balance+correct", "only_log_rank"]

    "min_samples_leaf": [10, 20] if short else [10, 25],
    "max_features": [0.5] if short else [0.7, "sqrt"],
    "n_jobs": [10]
}

# BOOST_param_grid = {
#     "size_sample": [0.5] if short else [0.7],
#     "n_estimators": [15] if short else [15], 
#     "ens_metric_name": ["ibs"] if short else ["CI","ibs"],
#     "depth": [15, 25],
#     "mode_wei": ['exp'] if short else ['square','exp'],
#     # "woe" : [],
#     "criterion": ["logrank"] if short else ["peto", "tarone-ware", "wilcoxon", "logrank"],
#     "min_samples_leaf": [5, 10, 20],
#     "max_features": [0.7] if short else [0.7, "sqrt"],
#     "aggreg_func": ['wei'] if short else ['wei', 'mean']
# }

BOOST_param_grid = {
    "size_sample": [0.7] if short else [0.7],
    "n_estimators": [20] if short else [20],  # new 50
    "ens_metric_name": ["CI"] if short else ["bic", "CI", "ibs"],  # ["CI", "ibs"],
    "depth": [5],  # 15
    "mode_wei": ['square'] if short else ['linear', 'exp', 'square', "sigmoid", "softmax"],
    "criterion": ["weights", "wilcoxon"] if short else ["confident", "weights",
                                                        "peto", "tarone-ware", "wilcoxon", "logrank"],
    "min_samples_leaf": [10] if short else [10, 25],
    "max_features": [0.7] if short else [0.7, "sqrt"],
    "aggreg_func": ['wei'] if short else ['wei', 'mean'],
    "leaf_model": ["base_fast"] if short else ["base_fast", 'wei_survive'],  # , "wei_survive"
    "all_weight": [True],
    "balance": [None, "balance", "only_log_rank"],
    "with_arc": [False],
    "n_jobs": [2]
}

SUMBOOST_param_grid = BOOST_param_grid.copy()
SUMBOOST_param_grid["learning_rate"] = [1.0, 0.2]

PROBOOST_param_grid = BOOST_param_grid.copy()
del PROBOOST_param_grid["mode_wei"]

WUHAN_PARAMS = {
    "TREE": CRAID_param_grid,
    "BSTR": BSTR_param_grid,
    "BOOST": BOOST_param_grid,
    "SUMBOOST": SUMBOOST_param_grid,
    "PROBOOST": PROBOOST_param_grid,
    "IBSBOOST": PROBOOST_param_grid,
    "IBSPROBOOST": PROBOOST_param_grid
}
