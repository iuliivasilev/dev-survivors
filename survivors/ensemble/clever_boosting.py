import numpy as np

from .. import metrics as metr
from ..tree import CRAID
from .. import constants as cnt
from .boosting import BoostingCRAID


class IBSCRAID(CRAID):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ibs_leaf = None

    def set_ibs_by_leaf(self, X, y):
        numbs = self.predict(X, target="numb").astype("int")
        sf = self.predict_at_times(X, self.bins, mode="surv")
        ibs_v = metr.ibs_WW(y, y, sf, self.bins, axis=0)

        counts = np.bincount(numbs)
        self.ibs_leaf = np.bincount(numbs, weights=ibs_v)
        self.ibs_leaf[counts > 0] /= counts[counts > 0]

    def get_ibs_by_leaf(self, X, divide=False):
        numbs = self.predict(X, target="numb").astype("int")
        return self.ibs_leaf[numbs]

    def fit(self, X, y):
        super().fit(X, y)
        self.set_ibs_by_leaf(X, y)


class IBSCleverBoostingCRAID(BoostingCRAID):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IBSCleverBoostingCRAID"

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
        self.l_ibs = []
        self.l_weights = []
        self.update_params()

        for i in range(self.n_estimators):
            x_sub = self.X_train.sample(n=self.size_sample, weights=self.weights,
                                        replace=self.bootstrap, random_state=i)
            x_oob = self.X_train.loc[self.X_train.index.difference(x_sub.index), :]
            #             print("UNIQUE:", np.unique(x_sub.index).shape[0])
            x_sub = x_sub.reset_index(drop=True)
            X_sub_tr, y_sub_tr = cnt.pd_to_xy(x_sub)
            if self.weighted_tree:
                X_sub_tr["weights_obs"] = self.weights[x_sub['ind_start']]  # self.weights

            model = IBSCRAID(features=self.features, random_state=i, **self.tree_kwargs)
            model.fit(X_sub_tr, y_sub_tr)

            wei_i, betta_i = self.count_model_weights(model, X_sub_tr, y_sub_tr)
            self.add_model(model, x_oob, wei_i, betta_i)
            self.update_weight(x_sub['ind_start'], wei_i)

    def predict(self, x_test, aggreg=True, **kwargs):
        res = []
        weights = []
        for i in range(len(self.models)):
            res.append(self.models[i].predict(x_test, **kwargs))
            weights.append(self.models[i].get_ibs_by_leaf(x_test))

        res = np.array(res)
        weights = np.vstack(weights).T
        if aggreg:
            res = self.get_aggreg(res, weights)
        return res

    def predict_at_times(self, x_test, bins, aggreg=True, mode="surv"):
        res = []
        weights = []
        for i in range(len(self.models)):
            res.append(self.models[i].predict_at_times(x_test, bins=bins,
                                                       mode=mode))
            weights.append(self.models[i].get_ibs_by_leaf(x_test))

        res = np.array(res)
        weights = np.vstack(weights).T
        if aggreg:
            res = self.get_aggreg(res, weights)
        return res

    def count_model_weights(self, model, X_sub, y_sub):
        if self.all_weight:
            X_sub = self.X_train
            y_sub = self.y_train
        pred_sf = model.predict_at_times(X_sub, bins=self.bins, mode="surv")
        #         m = metr.IBS_DICT.get(self.ens_metric_name.upper(), metr.ibs)
        m = metr.ibs_WW
        wei = m(self.y_train, y_sub, pred_sf, self.bins, axis=0) + 1e-15
        return wei, np.mean(wei)

    def update_weight(self, index, wei_i):
        if len(self.models) > 1:
            self.weights = 1 / (1 / self.weights + 1 / wei_i)
        else:
            self.weights = wei_i

    def get_aggreg(self, x, wei=None):
        if self.aggreg_func == 'median':
            return np.median(x, axis=0)
        elif self.aggreg_func == "obs_wei":
            wei = 1 / wei * 1 / np.sum(1 / wei)
            return np.sum((x.T * wei).T, axis=0)
        return np.mean(x, axis=0)

    def tolerance_find_best(self, ens_metric_name="bic"):
        pass
