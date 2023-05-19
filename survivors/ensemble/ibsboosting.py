from .boosting import BoostingCRAID
from .. import metrics as metr
import numpy as np


class IBSBoostingCRAID(BoostingCRAID):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IBSBoostingCRAID"

    def count_model_weights(self, model, X_sub, y_sub):
        if self.all_weight:
            X_sub = self.X_train
            y_sub = self.y_train

        pred_sf = model.predict_at_times(X_sub, bins=self.bins, mode="surv")

        wei = metr.ibs(self.y_train, y_sub, pred_sf, self.bins, axis=0)
        betta = np.mean(wei)
        return wei, betta

    def update_weight(self, index, wei_i):
        if self.all_weight:
            self.weights = self.weights + wei_i
        else:
            self.weights[index] = (self.weights[index] + wei_i)
        self.weights = (self.weights - self.weights.min())
        # self.weights = (self.weights - self.weights.min()) / (self.weights.max() - self.weights.min())

    def get_aggreg(self, x):
        if self.aggreg_func == 'median':
            return np.median(x, axis=0)
        if self.aggreg_func == 'moda':
            res = np.median(x > 0.5, axis=0)
            return res
        if self.aggreg_func == 'complex':
            # a = np.max(x, axis=0)
            b = np.min(x, axis=0)
            res = np.mean(x, axis=0)
            # res[a == 1] = 1
            res[res < 0.01] = 0
            return res
        elif self.aggreg_func == 'wei':
            inv_wei = 1 / np.array(self.bettas)
            wei = inv_wei / sum(inv_wei)
            return np.sum((x.T * wei).T, axis=0)
        return np.mean(x, axis=0)


class IBSProbBoostingCRAID(BoostingCRAID):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IBSProbBoostingCRAID"

    def count_model_weights(self, model, X_sub, y_sub):
        if self.all_weight:
            X_sub = self.X_train
            y_sub = self.y_train

        pred_sf = model.predict_at_times(X_sub, bins=self.bins, mode="surv")

        ibs_vals = metr.ibs(self.y_train, y_sub, pred_sf, self.bins, axis=0)
        ibs_scaled = (ibs_vals - np.mean(ibs_vals))  # first scheme
        # ibs_scaled = (ibs_vals - np.mean(ibs_vals)) / np.std(ibs_vals) # second scheme
        wei = 1 / (1 + np.exp(-ibs_scaled))
        betta = np.mean(wei)  # np.mean(ibs_vals)/np.std(ibs_vals)
        return wei, betta

    def update_weight(self, index, wei_i):
        if self.all_weight:
            self.weights = self.weights * wei_i
        else:
            self.weights[index] = (self.weights[index] * wei_i)
        # self.weights = self.weights / np.sum(self.weights)  # second scheme

    def get_aggreg(self, x):
        if self.aggreg_func == 'median':
            return np.median(x, axis=0)
        if self.aggreg_func == 'moda':
            res = np.median(x > 0.5, axis=0)
            return res
        if self.aggreg_func == 'complex':
            # a = np.max(x, axis=0)
            b = np.min(x, axis=0)
            res = np.mean(x, axis=0)
            # res[a == 1] = 1
            res[res < 0.01] = 0
            return res
        elif self.aggreg_func == 'wei':
            inv_wei = 1 / np.array(self.bettas)
            wei = inv_wei / sum(inv_wei)
            return np.sum((x.T * wei).T, axis=0)
        return np.mean(x, axis=0)
