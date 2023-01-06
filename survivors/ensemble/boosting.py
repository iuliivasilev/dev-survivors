# import pandas as pd
import numpy as np

from .. import metrics as metr
from ..tree import CRAID
from .. import constants as cnt
from .base_ensemble import BaseEnsemble, FastBaseEnsemble


def loss_func(var, mode='linear'):
    D = var.max()
    if D == 0:
        print("ALL LOSSES ARE NULL")
        var = np.full(var.shape[0], 1e-5)
        D = 1
    if mode == 'linear':
        return var/D
    elif mode == 'square':
        return (var/D)**2
    elif mode == 'exp':
        return 1.0 - np.exp(-var/D)
    elif mode == 'softmax':
        sm = np.exp(var/D)
        return sm/np.sum(sm)
    elif mode == 'sigmoid':
        # corr = np.exp((var - np.mean(var)) / np.std(var))
        return 1 / (1 + np.exp(-var/D))
    return None


def count_weight(losses, mode='linear', pred_wei=None):
    li = loss_func(losses, mode=mode)
    if pred_wei is None:
        l_mean = li.mean()  # (li * pred_wei).sum()/pred_wei.sum()
    else:
        norm_wei = pred_wei/pred_wei.sum()
        l_mean = (li * norm_wei).sum()

    betta = l_mean/(1.0 - l_mean)
    new_wei = betta ** (1 - li)
    return new_wei, betta


class BoostingCRAID(FastBaseEnsemble):
    """
    Adaptive boosting ensemble of survival decision tree.
    On each iteration probabilities of observations change by scheme.

    Attributes
    ----------
    mode_wei : str
        Type of weighting scheme
    kwargs : dict
        Parameters for building base ensemble (look at BaseEnsemble)
    
    weights : array-like 
        Current weights for observation
    bettas : list
        Confidence coeff for each base model 
    l_weights : list
        List of weights array for each observation

    Methods
    -------
    fit : build ensemble with X, y data
    count_model_weights : count weights according ibs and mode scheme
    add_model : add model, weights and bettas
        
    References
    ----------
    .. [1] Drucker H. Improving regressors using boosting techniques 
            //ICML. – 1997. – Т. 97. – С. 107-115.
    """
    def __init__(self, mode_wei="linear", weighted_tree=True, all_weight=False, **kwargs):
        self.name = "BoostingCRAID"
        self.mode_wei = mode_wei
        self.weights = None
        self.weighted_tree = weighted_tree
        self.all_weight = all_weight
        if self.weighted_tree:
            kwargs["weights_feature"] = "weights_obs"
        super().__init__(**kwargs)
        
    def fit(self, X, y):
        self.features = X.columns
        X = X.reset_index(drop=True)
        X[cnt.CENS_NAME] = y[cnt.CENS_NAME].astype(np.int32)
        X[cnt.TIME_NAME] = y[cnt.TIME_NAME].astype(np.float32)
        
        self.X_train = X
        self.X_train["ind_start"] = self.X_train.index
        self.y_train = y
        
        self.weights = np.ones(self.X_train.shape[0], dtype=float)
        self.bettas = []
        self.l_weights = []
        self.update_params()
        
        for i in range(self.n_estimators):
            x_sub = self.X_train.sample(n=self.size_sample, weights=self.weights,
                                        replace=self.bootstrap, random_state=i)
            x_oob = self.X_train.loc[self.X_train.index.difference(x_sub.index), :]

            x_sub = x_sub.reset_index(drop=True)
            X_sub_tr, y_sub_tr = cnt.pd_to_xy(x_sub)
            if self.weighted_tree:
                X_sub_tr["weights_obs"] = self.weights[x_sub['ind_start']]

            model = CRAID(features=self.features, random_state=i, **self.tree_kwargs)
            model.fit(X_sub_tr, y_sub_tr)
            
            wei_i, betta_i = self.count_model_weights(model, X_sub_tr, y_sub_tr)
            self.add_model(model, x_oob, wei_i, betta_i)
            self.update_weight(x_sub['ind_start'], wei_i)
            
            # print(betta_i)
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
        print('fitted:', len(self.models), 'models.')
    
    # def count_model_weights(self, model):
    #     pred_surv = model.predict_at_times(self.X_train, bins = self.bins, mode = "surv")
    #     losses = metr.ibs(self.y_train, self.y_train, pred_surv, self.bins, axis = 0)
    #     wei, betta = count_weight(losses, mode = self.mode_wei)
    #     return wei, betta
    
    # def update_weight(self, wei_i):
    #     self.weights[self.X_train.index] *= wei_i
    
    def count_model_weights(self, model, X_sub, y_sub):
        if self.all_weight:
            X_sub = self.X_train
            y_sub = self.y_train
        pred_sf = model.predict_at_times(X_sub, bins=self.bins, mode="surv")
        losses = metr.ibs(self.y_train, y_sub, pred_sf, self.bins, axis=0)

        # pred_wei = self.weights[X_sub['ind_start']]  # weight losses through previous

        pred_wei = None
        if self.all_weight:
            pred_wei = self.weights
        wei, betta = count_weight(losses, mode=self.mode_wei, pred_wei=pred_wei)
        return wei, betta
    
    def update_weight(self, index, wei_i):
        if self.all_weight:
            self.weights = self.weights * wei_i
        else:
            self.weights[index] = (self.weights[index] * wei_i)
        # ws = self.weights.sum()
        # if ws == 0.0 or ws != ws:
        #     self.weights = np.ones(self.X_train.shape[0], dtype=float)
        self.weights = self.weights/self.weights.sum()
    
    def add_model(self, model, x_oob, wei_i, betta_i):
        self.bettas.append(betta_i)
        self.l_weights.append(wei_i)
        super().add_model(model, x_oob)
    
    def select_model(self, start, end):
        self.bettas = self.bettas[start:end]
        self.l_weights = self.l_weights[start:end]
        super().select_model(start, end)
    
    def get_aggreg(self, x):
        """
        Overload function of response aggregation.
        Have median, weight (weighted sum with bettas) and mean mode.

        Parameters
        ----------
        x : Pandas dataframe
            Contain input features of events.

        Returns
        -------
        float
        """
        if self.aggreg_func == 'median':
            return np.median(x, axis=0)
        elif self.aggreg_func == 'wei':
            inv_wei = -1*np.log(self.bettas)
            wei = inv_wei/sum(inv_wei)
            return np.sum((x.T*wei).T, axis=0)
        return np.mean(x, axis=0)


class SumBoostingCRAID(BoostingCRAID):
    def __init__(self, learning_rate=0.1, **kwargs):
        self.name = "SumBoostingCRAID"
        self.learning_rate = learning_rate
        super().__init__(**kwargs)

    def update_weight(self, index, wei_i):
        if self.all_weight:
            self.weights = self.weights + self.learning_rate*wei_i
        else:
            self.weights[index] = (self.weights[index] + self.learning_rate*wei_i)
        self.weights = self.weights/self.weights.sum()
