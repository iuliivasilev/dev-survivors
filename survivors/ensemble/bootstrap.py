# import pandas as pd
import numpy as np

from ..tree import CRAID
from .. import constants as cnt
from .base_ensemble import BaseEnsemble


class BootstrapCRAID(BaseEnsemble):
    """
    Bootstrap aggregation (Bagging) ensemble of survival decision tree.
    On each iteration probabilities of observations change by scheme.

    Attributes
    ----------
    kwargs : dict
        Parameters for building base ensemble (look at BaseEnsemble)

    Methods
    -------
    fit : build ensemble with X, y data
    """
    def __init__(self, **kwargs):
        self.name = "BootstrapCRAID"
        super().__init__(**kwargs)
    
    def fit(self, X, y):
        self.features = X.columns
        X = X.reset_index(drop=True)
        X[cnt.CENS_NAME] = y[cnt.CENS_NAME].astype(np.int32)
        X[cnt.TIME_NAME] = y[cnt.TIME_NAME].astype(np.float32)
        
        self.X_train = X
        self.y_train = y
        self.update_params()
        
        for i in range(self.n_estimators):
            x_sub = self.X_train.sample(n=self.size_sample, replace=self.bootstrap, random_state=i)
            x_oob = self.X_train.loc[self.X_train.index.difference(x_sub.index), :]
            
            x_sub = x_sub.reset_index(drop=True)
            X_sub_tr, y_sub_cr = cnt.pd_to_xy(x_sub)
            
            model = CRAID(features=self.features, random_state=i, **self.tree_kwargs)
            model.fit(X_sub_tr, y_sub_cr)         

            self.add_model(model, x_oob)
            self.ens_metr[i] = self.score_oob()
            if not (self.tolerance) and i > 0:
                print(f"METRIC: {self.ens_metr[i-1]} -> +1 model METRIC: {self.ens_metr[i]}")
                if self.descend_metr:
                    stop = self.ens_metr[i-1] < self.ens_metr[i]
                else:
                    stop = self.ens_metr[i-1] > self.ens_metr[i]
                if stop:
                    self.select_model(0, len(self.models)-1)
                    break
        
        if self.tolerance:
            self.tolerance_find_best()
        print(f"fitted: {len(self.models)} models.")
