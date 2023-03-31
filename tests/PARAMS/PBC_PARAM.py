short = False

CRAID_param_grid = {
    "depth": [10, 15],
    "balance": [None, "balance", "balance+correct", "balance+weights"],
    "criterion": ["peto"] if short else ["confident_weights"],  # ["weights"],  # ["peto", "tarone-ware", "wilcoxon", "logrank"],
    "min_samples_leaf": [5] if short else [5],
    'cut': [True, False],
    "woe": [False],  # if short else [True, False],
    "signif": [0.05] if short else [0.05, 0.1],
    "max_features": [1.0],
    # "n_jobs": [8]
}

BSTR_param_grid = {
    "size_sample": [0.7], 
    "n_estimators": [10] if short else [10, 30],
    "depth": [15],
    "ens_metric_name": ["roc"] if short else ["conc", "ibs", "roc"],
    # "woe" : [], 
    "criterion": ["peto"] if short else ["peto", "tarone-ware", "wilcoxon", "logrank"], 
    "min_samples_leaf": [1, 5, 10],
    "max_features": [0.3] if short else [0.3, "sqrt"]
}

# BOOST_param_grid = {
#     "aggreg_func": ['wei'] if short else ['wei', 'mean'],
#     "criterion": ["logrank"] if short else ["peto", "tarone-ware", "wilcoxon", "logrank"],
#     "depth": [15],
#     "ens_metric_name": ["ibs"] if short else ["conc","ibs"],
#     "max_features": [0.3] if short else [0.3], #"sqrt"],
#     "min_samples_leaf": [1] if short else [1, 5, 15],
#     "mode_wei": ['square', 'exp'] if short else ['square'],#'exp'],
#     "n_estimators": [15] if short else [10, 15, 25],
#     "size_sample": [0.5] if short else [0.5, 0.7],
#     # "woe" : [],
# }

BOOST_param_grid = {
    "aggreg_func": ['wei'] if short else ['wei', 'mean'],
    "criterion": ["confident", "weights"] if short else ["confident", "confident_weights", "weights",
                                            "logrank", "peto", "tarone-ware", "wilcoxon"],
    "depth": [5],  # old 10,
    "ens_metric_name": ["roc"] if short else ["bic", "ibs", "roc"],  # ["ibs", "roc"],
    "max_features": ["sqrt"] if short else ["sqrt"],  # 0.3
    "min_samples_leaf": [1] if short else [1, 10],  # new [10]
    "mode_wei": ['square', 'exp'] if short else ['exp', 'square', "sigmoid", "softmax"],
    "n_estimators": [15] if short else [30],  # new 50 -> [10, 15, 25],
    "size_sample": [0.5] if short else [0.5, 0.7],
    "all_weight": [False],
    "leaf_model": ["base_fast"] if short else ["base_fast"],
    "balance": [None, "balance", "balance+weights"],
    "with_arc": [True, False],
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
    "PROBOOST": PROBOOST_param_grid
}
