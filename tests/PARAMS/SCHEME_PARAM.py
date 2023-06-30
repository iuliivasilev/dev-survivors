CRAID_param_grid = {
    "depth": [10],
    "balance": [None],  # [None, "balance", "balance+correct", "only_log_rank"]
    "criterion": ["maxcombo", "peto", "tarone-ware", "wilcoxon", "logrank"],
    "min_samples_leaf": [5, 10, 25],
    "leaf_model": ["base_fast"],
    'cut': [True, False],
    "woe": [True],
    "signif": [0.05, 1.0],
    "max_features": [1.0],
    "n_jobs": [10]
}

BSTR_param_grid = {
    "size_sample": [0.7, 1.0],  # 0.7
    "n_estimators": [50],
    "depth": [10],
    "ens_metric_name": ["IBS"],  # ["bic", "roc", "ibs"], , "IBS_WW", "IBS_REMAIN"
    "criterion": ["maxcombo", "peto", "tarone-ware", "wilcoxon", "logrank"],
    "leaf_model": ["base_fast", "base_zero_after"],  # "base_zero_after", "base_fast"
    "balance": [None],  #, "balance", "only_log_rank"],  # [None, "balance", "balance+correct", "only_log_rank"]

    "min_samples_leaf": [0.05, 0.001],
    "max_features": [0.3, "sqrt"],
    "n_jobs": [5]
}

SCHEME_PARAMS = {
    "TREE": CRAID_param_grid,
    "BSTR": BSTR_param_grid
}
