short = False

CRAID_param_grid = {
    "depth": [10, 15],
    "criterion": ["peto"] if short else ["peto", "tarone-ware", "wilcoxon", "logrank"],
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
    "criterion": ["weights"] if short else ["confident", "confident_weights", "weights",
                                            "logrank", "peto", "tarone-ware", "wilcoxon"],
    "depth": [10],  # 25],
    "ens_metric_name": ["roc"] if short else ["ibs", "roc"],
    "max_features": ["sqrt"] if short else ["sqrt"],  # 0.3
    "min_samples_leaf": [1] if short else [1, 10],  # 15],
    "mode_wei": ['square', 'exp'] if short else ['exp', 'square', "sigmoid", "softmax"],
    "n_estimators": [15] if short else [25],  # [10, 15, 25],
    "size_sample": [0.5] if short else [0.5, 0.7],
    "all_weight": [False],
    "leaf_model": ["base_fast"] if short else ["wei_survive", "base_fast"],
    # "leaf_model": ["base"] if short else ["base_fast", "wei_survive"],
    "n_jobs": [2],
    # ONLY SUM BOOSTING
    "learning_rate": [1.0, 0.2]
}

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
    "SUMBOOST": BOOST_param_grid
}
