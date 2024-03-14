short = False

CRAID_param_grid = {
    "depth": [10],
    "balance": [None],  # [None, "balance", "balance+correct", "only_log_rank"]
    "criterion": ["peto"] if short else ["confident", "peto", "tarone-ware", "wilcoxon", "logrank"],
    "min_samples_leaf": [5] if short else [1, 10],
    "leaf_model": ["base_fast"] if short else ["base_fast"],
    'cut': [True, False],
    "woe": [True],  # if short else [True, False],
    "signif": [0.05] if short else [0.05, 1.0],
    "max_features": [1.0],
    # "n_jobs": [8]
}

BSTR_param_grid = {
    "size_sample": [0.5, 0.7],  # 0.7
    "n_estimators": [10] if short else [30],
    "depth": [10],
    "ens_metric_name": ["IBS_REMAIN"],  # ["bic", "roc", "ibs"],
    "criterion": ["peto"] if short else ["confident", "peto", "tarone-ware", "wilcoxon", "logrank"],
    "leaf_model": ["base"] if short else ["base_fast"],
    "balance": [None],  # [None, "balance", "balance+correct", "only_log_rank"]
    "min_samples_leaf": [1, 10],
    "max_features": [0.3] if short else ["sqrt", 0.5]
}

# BOOST_param_grid = {
#     "aggreg_func": ['wei'] if short else ['wei', 'mean'],
#     "criterion": ["logrank"] if short else ["peto", "tarone-ware", "wilcoxon", "logrank"],
#     "depth": [15],
#     "ens_metric_name": ["ibs"] if short else ["CI","ibs"],
#     "max_features": [0.3] if short else [0.3], #"sqrt"],
#     "min_samples_leaf": [1] if short else [1, 5, 15],
#     "mode_wei": ['square', 'exp'] if short else ['square'],#'exp'],
#     "n_estimators": [15] if short else [10, 15, 25],
#     "size_sample": [0.5] if short else [0.5, 0.7],
#     # "woe" : [],
# }

BOOST_param_grid = {
    "aggreg_func": ['wei'] if short else ['wei', 'mean'],
    "criterion": ["confident", "weights"] if short else ["confident", "weights",
                                            "logrank", "peto", "tarone-ware", "wilcoxon"],
    "depth": [5],  # old 10,
    "ens_metric_name": ["roc"] if short else ["bic", "ibs"],  # ["ibs", "roc"],
    "max_features": ["sqrt"] if short else ["sqrt"],  # 0.3
    "min_samples_leaf": [1] if short else [1, 10],  # new [10]
    "mode_wei": ['square', 'exp'] if short else ['exp', 'square', "sigmoid", "softmax"],
    "n_estimators": [15] if short else [30],  # new 50 -> [10, 15, 25],
    "size_sample": [0.5] if short else [0.5, 0.7],
    "all_weight": [True],
    "leaf_model": ["base_fast"] if short else ["base_fast", "wei_survive"],
    "balance": [None, "balance", "only_log_rank"],
    "with_arc": [False],
    # "leaf_model": ["base"] if short else ["base_fast", "wei_survive"],
    "n_jobs": [2]
}

SUMBOOST_param_grid = BOOST_param_grid.copy()
SUMBOOST_param_grid["learning_rate"] = [1.0, 0.2]

PROBOOST_param_grid = BOOST_param_grid.copy()
del PROBOOST_param_grid["mode_wei"]

# BOOST_param_grid_error = {'aggreg_func': ['wei'],
#                     'criterion': ['weights'],
#                     'depth': [15], 'ens_metric_name': ['ibs'],
#                     'leaf_model': ['only_hazard'],
#                     'max_features': ['sqrt'], 'min_samples_leaf': [5],
#                     'mode_wei': ['exp'], 'n_estimators': [25],
#                     'n_jobs': [1], 'size_sample': [0.7], 'weighted_tree': [True]}

PBC_PARAMS = {
    "TREE": CRAID_param_grid,
    "BSTR": BSTR_param_grid,
    "BOOST": BOOST_param_grid,
    "SUMBOOST": SUMBOOST_param_grid,
    "PROBOOST": PROBOOST_param_grid,
    "IBSBOOST": PROBOOST_param_grid,
    "IBSPROBOOST": PROBOOST_param_grid
}
