CRAID_param_grid = {
    "depth": [10],
    "balance": [None],  # [None, "balance", "balance+correct", "only_log_rank"]
    "criterion": ["confident", "peto", "tarone-ware", "wilcoxon", "logrank"],
    "min_samples_leaf": [5, 10, 25],
    "leaf_model": ["base_fast"],
    'cut': [True, False],
    "woe": [True],
    "signif": [0.05, 1.0],
    "max_features": [1.0],
    "n_jobs": [10]
}

BSTR_param_grid = {
    "size_sample": [0.5, 0.7],  # 0.7
    "n_estimators": [30],
    "depth": [10],
    "ens_metric_name": ["bic", "roc", "ibs"],
    "criterion": ["confident", "peto", "tarone-ware", "wilcoxon", "logrank"],
    "leaf_model": ["base_fast"],
    "balance": [None],  # [None, "balance", "balance+correct", "only_log_rank"]

    "min_samples_leaf": [5, 10, 25],
    "max_features": [0.7, "sqrt"],
    "n_jobs": [10]
}

SCHEME_PARAMS = {
    "TREE": CRAID_param_grid,
    "BSTR": BSTR_param_grid
}
