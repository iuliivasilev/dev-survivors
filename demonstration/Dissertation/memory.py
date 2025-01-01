import pandas as pd
import numpy as np
import os
import time
import copy
import pathlib, tempfile

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()

custom_params = {"axes.spines.right": False, 'grid.color': 'lightgray', 'axes.grid': True, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

from graphviz import Digraph
from joblib import Parallel, delayed
from scipy import stats

from survivors import metrics as metr
from survivors import constants as cnt
from survivors import criteria as crit
from numba import njit, jit, int32, float64
from lifelines import KaplanMeierFitter, NelsonAalenFitter

import survivors.datasets as ds

import cProfile
import pstats


from sklearn.model_selection import train_test_split, StratifiedKFold
from survivors.experiments.grid import generate_sample, prepare_sample, count_metric

# X, y, features, categ, sch_nan = ds.load_actg_dataset()
X, y, features, categ, sch_nan = ds.load_smarto_dataset()
# X, y, features, categ, sch_nan = ds.load_wuhan_dataset()

qs = np.quantile(y["time"], np.linspace(0.2, 0.8, 4))
time_discr = np.searchsorted(qs, y["time"])

discr = np.char.add(time_discr.astype(str), y["cens"].astype(str))
X_TR, X_HO = train_test_split(X, stratify=discr, test_size=0.33, random_state=42)
X_tr, y_tr, X_HO, y_HO, bins_HO = prepare_sample(X, y, X_TR.index, X_HO.index)

from memory_profiler import memory_usage
from survivors.tree import CRAID
import sys

p = {'balance': None, 'categ': categ, 'criterion': 'wilcoxon', 'cut': False, 'depth': 10, 
     'ens_metric_name': 'IBS_REMAIN', 'l_reg': 0.0, 'leaf_model': 'base_zero_after', 
     'max_features': 0.9, 'min_samples_leaf': 0.01, 'n_jobs': 1, 'n_jobs_loop': 1, 'signif': 1, 'woe': True}

def train_model():
    tree = CRAID(**p)
    tree.fit(X_tr, y_tr)

if __name__ == "__main__":
    train_model()