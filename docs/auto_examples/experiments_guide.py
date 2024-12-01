"""
=================
Experiments guide
=================

"""

# Author: Iulii Vasilev <iuliivasilev@gmail.com>
#
# License: BSD 3 clause

# %%
# First, we will import modules
#

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# %%
# ### Create Experiments instance
# 
# 1. The dataset is represented as a tuple:
#     - X (pd.DataFrame): Feature space
#     - y (structured np.ndarray): Target variables
#     - features (list): Covariates
#     - categ (list): Categorical covariates
# 2. Metrics are defined according to the metrics module (keys of METRIC_DICT dictionary)
# 3. Creating Experiments specify:
#     - folds (int): Quantity of cross-validate folds
#     - mode (str): Validation strategy.
#     
# Available modes: "CV", "CV+HOLD-OUT", **"CV+SAMPLE"**, "TIME-CV".
# In **this** case, five-fold cross-validation is performed followed by training the best models on 20 different samples of the original data.
# The final quality is defined as the average quality over 20 samples.
# 
# Methods for adding metrics (**set_metrics**), metrics for selecting the best models (**add_metric_best**) are used to control the experiment.
#

from survivors.experiments import grid as exp
import survivors.datasets as ds

l_metrics = ["CI", "CI_CENS", "IBS", "IBS_REMAIN", "IAUC", "IAUC_WW_TI", "AUPRC"]
X, y, features, categ, _ = ds.load_pbc_dataset()
experim = exp.Experiments(folds=5, mode="CV+SAMPLE")
experim.set_metrics(l_metrics)
experim.add_metric_best("IBS_REMAIN")


# %%
# To add models, the **add_method** method is used with two parameters: model class and hyperparameter grid.
#

# %%
# ### Add models from external libraries
#
# Experiments support models from the external **scikit-survival** library. For each model a grid of hyperparameters is defined.
#

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.tree import SurvivalTree
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis

COX_param_grid = {
    'alpha': [100, 10, 1, 0.1, 0.01, 0.001],
    'ties': ["breslow"]
}

RSF_param_grid = {
    'n_estimators': [50],
    'max_depth': [None, 20],
    'min_samples_leaf': [0.001, 0.01, 0.1, 0.25],
    "random_state": [123]
}

ST_param_grid = {
    'max_depth': [None, 20, 30],
    'min_samples_leaf': [1, 10, 20],
    'max_features': [None, "sqrt"],
    "random_state": [123]
}

GBSA_param_grid = {
    'loss': ["coxph"],
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'n_estimators': [50],
    'max_depth': [20],
    'min_samples_leaf': [1, 10, 50, 100],
    'max_features': ["sqrt"],
    "random_state": [123]
}

CWGBSA_param_grid = {
    'loss': ["coxph"],
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'n_estimators': [30, 50],
    'subsample': [0.7, 1.0],
    'dropout_rate': [0.0, 0.1, 0.5],
    "random_state": [123]
}


# %%
#

experim.add_method(CoxPHSurvivalAnalysis, COX_param_grid)
experim.add_method(SurvivalTree, ST_param_grid)
experim.add_method(RandomSurvivalForest, RSF_param_grid)
experim.add_method(ComponentwiseGradientBoostingSurvivalAnalysis, CWGBSA_param_grid)
experim.add_method(GradientBoostingSurvivalAnalysis, GBSA_param_grid)


# %%
# ### Add embedded AFT models
#
# Some models of the external **lifelines** library (CoxPH, AFT, KaplanMeier, NelsonAalen) are also embedded in the library. 
# 
# Note that these models can be used in tree sheets to build stratified models.
# 
# To add your own model, you can use **LeafModel** wrapper from the external.leaf_model module.
#

from survivors.external import LogLogisticAFT, AFT_param_grid

experim.add_method(LogLogisticAFT, AFT_param_grid)


# %%
# ### Add models from "survivors"
# 
# Of course, the experiments support models from **survivors**:
# 
# 1. **CRAID**: a survival tree with weighted criteria, regularisation and complex non-parametric models.
# 2. **BootstrapCRAID**: ensemble of independent trees on bootstrap samples.
# 3. **ParallelBootstrapCRAID**: a parallel implementation of BootstrapCRAID.
# 4. **BoostingCRAID**: adaptive bootstrapping with weighting of observations by probability of hitting the next subsample and correction based on base model error.
# 

from survivors.tree import CRAID
from survivors.ensemble import ParallelBootstrapCRAID, BoostingCRAID

CRAID_param_grid = {
    "depth": [10],
    "criterion": ["wilcoxon", "logrank"],
    "l_reg": [0, 0.01, 0.1, 0.5],
    "min_samples_leaf": [0.05, 0.01, 0.001],
    "signif": [0.1, 1.0],
    "categ": [categ]
}

BSTR_param_grid = {
    "n_estimators": [50],
    "depth": [7],
    "size_sample": [0.3, 0.7],
    "l_reg": [0, 0.01, 0.1, 0.5],
    "criterion": ["peto", "tarone-ware", "wilcoxon", "logrank"],
    "min_samples_leaf": [0.05, 0.01],
    "ens_metric_name": ["IBS_REMAIN"],
    "max_features": ["sqrt"],
    "categ": [categ]
}

experim.add_method(CRAID, CRAID_param_grid)
experim.add_method(ParallelBootstrapCRAID, BSTR_param_grid)


# %%
# ### Run experiments
# 
# To run experiments, use the run_effective method with the source data and:
# - verbose (int): log printing parameter.
# - stratify_best (str/list): one or more hyperparameters on which to build independent best models (for each hyperparameter value).
# 
# ##### Execution may take some time.
# Experimental results can be obtained by calling methods:
# - get_result: dataframe of results at the cross-validation stage.
# - get_best_by_mode method: dataframe of model validation at 20 samples.
# 

experim.run_effective(X, y, verbose=0, stratify_best=[])
df_results = experim.get_result()

# %%
# 

df_validation = experim.get_best_by_mode()


# %%
# ### Visualization
# 
# For example, here are the **result table** values and **boxplot**.
# 
# For each metric, four columns are defined:
# - **\<metric>**: list of metric indicators on each of the 20 samples.
# - **\<metric>_mean**: the average value of the metric at the 20 samples.
# - **\<metric>_CV**: list of metric indicators on cross-validation.
# - **\<metric>_CV_mean**: the average value of the metric on cross-validation.
# 


df_validation[["METHOD"] + [m + "_mean" for m in l_metrics]]


# %%
# 

for m in l_metrics:
    fig, axs = plt.subplots(figsize=(8, 8))
    plt.title(m)
    plt.boxplot(df_validation[m], labels=df_validation['METHOD'], showmeans=True, vert=False)
    plt.show()
