CRAID_param_grid = {
    "depth": [15],  # 10
    "balance": [None],  # [None, "balance", "balance+correct", "only_log_rank"]
    "criterion": ["peto", "tarone-ware", "wilcoxon", "logrank"],  # "ibswei", "maxcombo",
    "min_samples_leaf": [0.05, 0.01, 0.001],
    "l_reg": [0, 0.01, 0.1, 0.5, 0.9],  # [0, 0.001, 0.01, 0.1, 0.2]
    "leaf_model": ["base_zero_after"],
    'cut': [False],  # [True, False]
    "woe": [True],
    "signif": [0.05, 0.1, 1.0],
    "max_features": [1.0],
    "n_jobs": [10]
}

BSTR_param_grid = {
    "size_sample": [0.7, 1.0],  # 0.7
    "n_estimators": [50],
    "depth": [10],  # 15
    "ens_metric_name": ["IBS"],  # ["bic", "roc", "ibs"], "IBS_WW", "IBS_REMAIN"
    "criterion": ["ibswei", "maxcombo", "peto", "tarone-ware", "wilcoxon", "logrank"],  # "ibswei",
    "leaf_model": ["base_fast", "base_zero_after"],  # ["base_fast", "base_zero_after"],  # "base_zero_after", "base_fast"
    "balance": [None, "only_log_rank"],  # "balance", "only_log_rank"],  # [None, "balance", "balance+correct", "only_log_rank"]

    "min_samples_leaf": [0.05, 0.01, 0.001],
    "max_features": [0.3, "sqrt"],
    "n_jobs": [5]
}

# BOOST_param_grid = {
#     "aggreg_func": ['wei', 'mean'],
#     "size_sample": [0.7, 1.0],  # 0.7
#     "n_estimators": [50],
#     "depth": [10],
#     "mode_wei": ['exp', 'square'],  # 'exp', 'square', "sigmoid", "softmax"],
#     "ens_metric_name": ["IBS"],  # ["bic", "roc", "ibs"], , "IBS_WW", "IBS_REMAIN"
#     "criterion": ["maxcombo", "peto", "tarone-ware", "wilcoxon", "logrank"],
#     "leaf_model": ["base_zero_after"],  # "base_zero_after", "base_fast"
#     "balance": [None],  #, "balance", "only_log_rank"],  # [None, "balance", "balance+correct", "only_log_rank"]
#
#     "min_samples_leaf": [0.05, 0.001],
#     "max_features": [0.3, "sqrt"],
#     "n_jobs": [5],
#     "with_arc": [False],
#     'weighted_tree': [False],
#     "all_weight": [False, True]
# }

BOOST_param_grid = {
    "aggreg_func": ['mean'],  # ['wei', 'mean'],
    "size_sample": [0.5, 1.0],  # 0.7
    "n_estimators": [50],
    "depth": [7],
    "l_reg": [0, 0.1, 0.9],
    "mode_wei": ['exp', 'square'],  # 'exp', 'square', "sigmoid", "softmax"],
    "ens_metric_name": ["IBS"],  # ["bic", "roc", "ibs"], , "IBS_WW", "IBS_REMAIN"
    "criterion": ["peto", "tarone-ware", "wilcoxon", "logrank"],
    "leaf_model": ["base_zero_after"],  # "base_zero_after", "base_fast"
    "balance": [None],  #, "balance", "only_log_rank"],  # [None, "balance", "balance+correct", "only_log_rank"]

    "min_samples_leaf": [0.01, 0.001],
    "max_features": [0.3, "sqrt"],
    "n_jobs": [5],
    "with_arc": [False],
    'weighted_tree': [False],
    "all_weight": [False]  # [False, True]
}

# CL_BOOST_param_grid = {
#     "aggreg_func": ['wei', 'mean'],
#     "size_sample": [0.7, 1.0],  # 0.7
#     "n_estimators": [50],
#     "depth": [10],
#     "ens_metric_name": ["IBS"],
#     "criterion": ["ibswei", "maxcombo", "peto", "tarone-ware", "wilcoxon", "logrank"],
#     "leaf_model": ["base_zero_after", "base_fast"],  # "base_zero_after", "base_fast"
#     "balance": [None],
#
#     "min_samples_leaf": [0.05, 0.001],
#     "max_features": [0.3, "sqrt"],
#     "n_jobs": [5],
#
#     "all_weight": [True],
#     "weighted_tree": [True]
# }

CL_BOOST_param_grid = {
    "aggreg_func": ["mean"],  # 'wei',
    "size_sample": [0.5, 1.0],  # 0.7
    "n_estimators": [50],
    "depth": [7],  # 10
    "l_reg": [0, 0.01, 0.1, 0.5, 0.9],  # [0, 0.001, 0.01, 0.05, 0.1],
    "ens_metric_name": ["IBS"],
    "criterion": ["peto", "tarone-ware", "wilcoxon", "logrank"],  # "symm_peto",
    "leaf_model": ["base_zero_after"],  # "base_fast"
    "balance": [None],

    "min_samples_leaf": [0.01, 0.001],  # 0.001,
    "max_features": [0.3, "sqrt"],
    "n_jobs": [5],

    "all_weight": [True],
    "weighted_tree": [False]
}

SCHEME_PARAMS = {
    "TREE": CRAID_param_grid,
    "BSTR": BSTR_param_grid,
    "BOOST": BOOST_param_grid,
    "CLEVERBOOST": CL_BOOST_param_grid
}
