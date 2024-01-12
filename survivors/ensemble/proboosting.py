import numpy as np

from .. import constants as cnt
from .boosting import BoostingCRAID


def loglikelihood_i(time, cens, sf, cumhf, bins):
    index_times = np.digitize(time, bins, right=True) - 1
    hf = np.hstack((cumhf[:, 0][np.newaxis].T, np.diff(cumhf)))
    sf_by_times = np.take_along_axis(sf, index_times[:, np.newaxis], axis=1)[:, 0] + 1e-10
    hf_by_times = (np.take_along_axis(hf, index_times[:, np.newaxis], axis=1)[:, 0] + 1e-10) ** cens
    return np.log(sf_by_times) + np.log(hf_by_times)


def values_to_hist(values):
    unq, idx = np.unique(values, return_inverse=True)
    # calculate the weighted frequencies of these indices
    freqs_idx = np.bincount(idx)
    # reconstruct the array of frequencies of the elements
    return freqs_idx[idx]


class ProbBoostingCRAID(BoostingCRAID):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ProbBoostingCRAID"

    def count_model_weights(self, model, X_sub, y_sub):
        if self.all_weight:
            X_sub = self.X_train
            y_sub = self.y_train

        pred_sf = model.predict_at_times(X_sub, bins=self.bins, mode="surv")
        pred_hf = model.predict_at_times(X_sub, bins=self.bins, mode="hazard")

        time_hist = values_to_hist(y_sub[cnt.TIME_NAME])
        lp_ti = np.log(time_hist / y_sub[cnt.TIME_NAME].shape)
        lp_xi_ti = np.log(1 / time_hist)
        likel = loglikelihood_i(y_sub[cnt.TIME_NAME], y_sub[cnt.CENS_NAME], pred_sf, pred_hf, self.bins)

        lp_xi = lp_ti + lp_xi_ti - likel
        wei = - np.exp(-lp_xi)
        betta = np.sum(likel)
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
        elif self.aggreg_func == 'wei':
            inv_wei = -1/np.array(self.bettas)
            wei = inv_wei/sum(inv_wei)
            return np.sum((x.T*wei).T, axis=0)
        return np.mean(x, axis=0)
