short = False

CRAID_param_grid = {
    "depth": [10],
    "balance": [None],  # [None, "balance", "balance+correct", "only_log_rank"]
    "criterion": ["peto"] if short else ["confident", "peto", "tarone-ware", "wilcoxon", "logrank"],
    "min_samples_leaf": [10] if short else [5, 50],
    "leaf_model": ["base_fast"] if short else ["base_fast"],
    'cut': [True, False],
    "woe": [True],  # if short else [True, False],
    "signif": [0.05] if short else [0.05, 1.0],
    "max_features": [1.0],
    "n_jobs": [4]
}

BSTR_param_grid = {
    "size_sample": [0.5, 0.7],  # 0.7
    "n_estimators": [10] if short else [50],
    "depth": [10],
    "ens_metric_name": ["bic", "roc", "ibs"],
    "criterion": ["peto"] if short else ["confident",
                                         "peto", "tarone-ware", "wilcoxon", "logrank"],
    "leaf_model": ["base_fast"] if short else ["base_fast"],
    "balance": [None],  # [None, "balance", "balance+correct", "only_log_rank"]

    "min_samples_leaf": [5, 50],
    "max_features": [0.2] if short else [0.5],  # ["sqrt"],
    "n_jobs": [40]
}

# BOOST_param_grid = {
#     "size_sample": [0.9],  # [0.5, 0.7],
#     "n_estimators": [100],  # old [30]
#     "ens_metric_name": ["ibs"] if short else ["bic", "ibs"],  # ["roc", "ibs"],  # ["CI", "ibs"]
#     "depth": [10] if short else [5, 10],  # new [5]
#     "mode_wei": ['exp'] if short else ['exp', 'square', "sigmoid", "softmax"],
#     "criterion": ["confident", "logrank"] if short else ["confident", "weights", "peto",
#                                                          "tarone-ware", "wilcoxon", "logrank"],
#     "min_samples_leaf": [5] if short else [5, 100],  # [20, 100]
#     "max_features": [0.4] if short else [0.5, 0.7],  # [0.2, 0.5],
#     "aggreg_func": ['wei'] if short else ['wei', 'mean'],
#     "leaf_model": ["base_fast"] if short else ["base_fast"],
#     "all_weight": [True, False],
#     "balance": [None, "balance", "balance+correct", "only_log_rank"],
#     "with_arc": [True, False],
#     "n_jobs": [2]
# }

BOOST_param_grid = {
    "size_sample": [0.7],  # [0.5, 0.7],
    "n_estimators": [50],  # old [30]
    "ens_metric_name": ["ibs"] if short else ["CI", "ibs"],  # ["roc", "ibs"],
    "depth": [10] if short else [10],  # new [5]
    "mode_wei": ['exp'] if short else ['exp', 'square', "sigmoid", "softmax"],
    "criterion": ["confident", "logrank"] if short else ["confident", "weights", "peto",
                                                         "tarone-ware", "wilcoxon", "logrank"],
    "min_samples_leaf": [5] if short else [5, 10],  # [20, 100]
    "max_features": [0.4] if short else [0.7, 0.9],  # [0.2, 0.5],
    "aggreg_func": ['wei'] if short else ['wei'],
    "leaf_model": ["base_fast"] if short else ["base_fast", "wei_survive"],
    "all_weight": [True],
    "balance": [None],  # [None, "balance", "only_log_rank"],
    "with_arc": [False],
    "n_jobs": [1]
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
    "PROBOOST": PROBOOST_param_grid,
    "IBSBOOST": PROBOOST_param_grid,
    "IBSPROBOOST": PROBOOST_param_grid
}
