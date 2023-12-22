import numpy as np

from .. import metrics as metr
from ..tree import CRAID
from .. import constants as cnt
from .boosting import BoostingCRAID
import matplotlib.pyplot as plt

from ..tree.stratified_model import KaplanMeierZeroAfter, NelsonAalen


def f_predict_survival_at_times(ch, bins=None):
    mean, std, prob = ch
    durs = np.random.normal(mean, std, 1000)
    events = np.random.binomial(1, prob, size=1000)
    events = events[durs > 0]
    durs = durs[durs > 0]

    survival = KaplanMeierZeroAfter()
    survival.fit(durs, events)
    if bins is None:
        bins = survival.timeline
    sf = survival.survival_function_at_times(bins)
    return sf


def f_predict_hazard_at_times(ch, bins=None):
    mean, std, prob = ch
    durs = np.random.normal(mean, std, 1000)
    events = np.random.binomial(1, prob, size=1000)
    events = events[durs > 0]
    durs = durs[durs > 0]

    hazard = NelsonAalen()
    hazard.fit(durs, events)
    if bins is None:
        bins = hazard.timeline
    hf = hazard.cumulative_hazard_at_times(bins)
    return hf

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
            x_sub = self.X_train.sample(n=self.size_sample,
                                        # weights=self.weights,
                                        replace=self.bootstrap, random_state=i)

            x_oob = self.X_train.loc[self.X_train.index.difference(x_sub.index), :]
            print(f"UNIQUE ({i}):{np.unique(x_sub.index).shape[0]}, DIST:", np.bincount(x_sub["cens"]))
            x_sub = x_sub.reset_index(drop=True)
            X_sub_tr, y_sub_tr = cnt.pd_to_xy(x_sub)
            if self.weighted_tree:
                X_sub_tr["weights_obs"] = self.weights[x_sub['ind_start']]

            model = CRAID(features=self.features, random_state=i, **self.tree_kwargs)
            model.fit(X_sub_tr, y_sub_tr)

            wei_i, betta_i = self.count_model_weights(model, X_sub_tr, y_sub_tr)
            self.add_model(model, x_oob, wei_i, betta_i)
            self.update_weight(x_sub['ind_start'], wei_i)

    def predict(self, x_test, aggreg=True, **kwargs):
        res = []
        weights = []
        for i in range(len(self.models)):
            res.append(self.models[i].predict(x_test, **kwargs))

        res = np.array(res)
        weights = None
        if aggreg:
            res = self.get_aggreg(res, weights)
        return res

    # def predict_at_times(self, x_test, bins, aggreg=True, mode="surv"):
    #     res = []
    #     weights = []
    #     for i in range(len(self.models)):
    #         res.append(self.models[i].predict_at_times(x_test, bins=bins,
    #                                                    mode=mode))
    #
    #     res = np.array(res)
    #     weights = None
    #     if aggreg:
    #         res = self.get_aggreg(res, weights)
    #         if mode == "surv":
    #             res[:, -1] = 0
    #             res[:, 0] = 1
    #     return res

    def predict_at_times(self, x_test, bins, aggreg=True, mode="surv"):
        res = []
        for i in range(len(self.models)):
            res.append(self.models[i].predict(x_test, target="ch")[np.newaxis, :])

        res = np.vstack(res)
        means, stds, probs = res.T
        mean = np.mean(means, axis=1)
        std = np.sqrt(np.sum(stds ** 2, axis=1)) / np.sqrt(stds.shape[1])
        # std = np.sqrt(np.sum(stds ** 2, axis=1)) / np.sqrt(stds.shape[1])
        #         std = np.sqrt(np.sum((1/stds.shape[1]*stds)**2, axis=1))*1/np.sqrt(stds.shape[1])
        #         std = np.mean(stds, axis=1)
        prob = np.mean(probs, axis=1)

        prepared = np.vstack([mean, std, prob]).T
        if mode == "surv":
            res = np.array(list(map(lambda x: f_predict_survival_at_times(x, bins), prepared)))
            res[:, -1] = 0
            res[:, 0] = 1
        else:
            res = np.array(list(map(lambda x: f_predict_hazard_at_times(x, bins), prepared)))
        return res

    def count_model_weights(self, model, X_sub, y_sub):
        # if self.all_weight:
        #     X_sub = self.X_train
        #     y_sub = self.y_train
        # pred_sf = model.predict_at_times(X_sub, bins=self.bins, mode="surv")
        #
        # ibs_sf = metr.auprc(self.y_train, y_sub, pred_sf, self.bins, axis=0)
        # betta = np.mean(ibs_sf)
        # wei = 1 - ibs_sf
        wei = np.ones(X_sub.shape[0])
        betta = 1
        return wei, abs(betta)

    def update_weight(self, index, wei_i):
        # self.weights += wei_i
        pass

    def get_aggreg(self, x, wei=None):
        if self.aggreg_func == 'median':
            return np.median(x, axis=0)
        elif self.aggreg_func == "wei":
            if wei is None:
                wei = np.array(self.bettas)
            wei = wei / np.sum(wei)
            return np.sum((x.T * wei).T, axis=0)
        elif self.aggreg_func == "argmean":
            wei = np.where(np.argsort(np.argsort(wei, axis=1), axis=1) > len(self.bettas) // 2, 1, 0)
            wei = wei / np.sum(wei, axis=1).reshape(-1, 1)
            return np.sum((x.T * wei).T, axis=0)
        elif self.aggreg_func == "argwei":
            wei = np.where(np.argsort(np.argsort(wei, axis=1), axis=1) > len(self.bettas) // 2,
                           1 / np.array(self.bettas), 0)
            wei = wei / np.sum(wei, axis=1).reshape(-1, 1)
            return np.sum((x.T * wei).T, axis=0)
        return np.mean(x, axis=0)

    def plot_curve(self, X_tmp, y_tmp, bins, label="", metric="ibs"):
        res = []
        metr_vals = []
        for i in range(len(self.models)):
            res.append(self.models[i].predict_at_times(X_tmp, bins=bins, mode="surv"))

            res_all = np.array(res)
            res_all = self.get_aggreg(res_all, np.array(self.bettas)[:i + 1])
            res_all[:, -1] = 0
            res_all[:, 0] = 1
            if metric == "ibs":
                metr_vals.append(metr.ibs_WW(self.y_train, y_tmp, res_all, bins))
            else:
                metr_vals.append(metr.auprc(self.y_train, y_tmp, res_all, bins))
        plt.plot(range(len(self.models)), metr_vals, label=label)

# class IBSCleverBoostingCRAID(BoostingCRAID):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.name = "IBSCleverBoostingCRAID"
#
#     def fit(self, X, y):
#         self.features = list(X.columns)
#         X = X.reset_index(drop=True).copy()
#         X[cnt.CENS_NAME] = y[cnt.CENS_NAME].astype(np.int32)
#         X[cnt.TIME_NAME] = y[cnt.TIME_NAME].astype(np.float32)
#
#         self.X_train = X.copy()
#         self.X_train["ind_start"] = self.X_train.index
#         self.y_train = y.copy()
#
#         self.weights = y[cnt.TIME_NAME].copy()
#         self.bettas = []
#         self.l_ibs = []
#         self.l_weights = []
#         self.update_params()
#         self.lr = 1
#
#         for i in range(self.n_estimators):
#             x_sub = self.X_train.sample(n=self.size_sample,
#                                         # weights=self.weights,
#                                         replace=self.bootstrap, random_state=i)
#
#             x_oob = self.X_train.loc[self.X_train.index.difference(x_sub.index), :]
#             # print(f"UNIQUE ({i}):{np.unique(x_sub.index).shape[0]}, DIST:", np.bincount(x_sub["cens"]))
#             x_sub = x_sub.reset_index(drop=True)
#             X_sub_tr, y_sub_tr = cnt.pd_to_xy(x_sub.copy())
#             y_sub_tr[cnt.TIME_NAME] = self.weights[x_sub['ind_start'].values].copy()
#
#             # plt.scatter(self.y_train["time"], self.weights, c=np.where(self.y_train["cens"], "orange", "blue"))
#             # plt.show()
#
#             model = CRAID(features=self.features.copy(), random_state=i, **self.tree_kwargs)
#             model.fit(X_sub_tr, y_sub_tr)
#
#             wei_i, betta_i = self.count_model_weights(model, X_sub_tr.copy(), y_sub_tr.copy())
#             self.add_model(model, x_oob, wei_i, betta_i)
#             self.update_weight(x_sub['ind_start'].values, wei_i)
#
#     def predict(self, x_test, aggreg=True, **kwargs):
#         res = []
#         weights = []
#         for i in range(len(self.models)):
#             res.append(self.models[i].predict(x_test, **kwargs))
#
#         res = np.array(res)
#         weights = None
#         if aggreg:
#             res = self.get_aggreg(res, weights)
#         return res
#
#     def predict_at_times(self, x_test, bins, aggreg=True, mode="surv"):
#         res = []
#         weights = []
#         for i in range(len(self.models)):
#             res.append(self.models[i].predict_at_times(x_test, bins=bins,
#                                                        mode=mode))
#
#         res = np.array(res)
#         weights = None
#         if aggreg:
#             res = self.get_aggreg(res, weights)
#             if mode == "surv":
#                 res[:, -1] = 0
#                 res[:, 0] = 1
#         return res
#
#     def count_model_weights(self, model, X_sub, y_sub):
#         if self.all_weight:
#             X_sub = self.X_train.copy()
#             y_sub = self.y_train.copy()
#
#         y_sub["time"] = self.weights[X_sub['ind_start'].values]
#         # pred_time = model.predict(X_sub, target="time")
#         pred_sf = model.predict_at_times(X_sub, bins=self.bins, mode="surv")
#
#         pred_time = self.bins[(pred_sf <= 0.5).argmax(axis=1)]
#         ibs_sf = metr.ibs_remain(self.y_train, y_sub, pred_sf, self.bins, axis=0)
#
#         wei = (y_sub["time"] - pred_time) * ibs_sf**2  # (2*y_sub["cens"] - 1) *
#         betta = self.lr / np.mean(ibs_sf)
#         return wei, abs(betta)
#
#     def update_weight(self, index, wei_i):
#         #         print(index, wei_i)
#         self.lr = 0.3
#         if self.all_weight:
#             self.weights -= wei_i * self.lr
#         else:
#             self.weights[index] -= wei_i * self.lr
#
#     def get_aggreg(self, x, wei=None):
#         if self.aggreg_func == 'median':
#             return np.median(x, axis=0)
#         elif self.aggreg_func == "wei":
#             if wei is None:
#                 wei = np.array(self.bettas)
#             wei = wei / np.sum(wei)
#             return np.sum((x.T * wei).T, axis=0)
#         elif self.aggreg_func == "argmean":
#             wei = np.where(np.argsort(np.argsort(wei, axis=1), axis=1) > len(self.bettas) // 2, 1, 0)
#             wei = wei / np.sum(wei, axis=1).reshape(-1, 1)
#             return np.sum((x.T * wei).T, axis=0)
#         elif self.aggreg_func == "argwei":
#             wei = np.where(np.argsort(np.argsort(wei, axis=1), axis=1) > len(self.bettas) // 2,
#                            1 / np.array(self.bettas), 0)
#             wei = wei / np.sum(wei, axis=1).reshape(-1, 1)
#             return np.sum((x.T * wei).T, axis=0)
#         return np.mean(x, axis=0)
#
#     def plot_curve(self, X_tmp, y_tmp, bins, label="", metric="ibs"):
#         res = []
#         metr_vals = []
#         for i in range(len(self.models)):
#             res.append(self.models[i].predict_at_times(X_tmp, bins=bins, mode="surv"))
#
#             res_all = np.array(res)
#             res_all = self.get_aggreg(res_all, np.array(self.bettas)[:i + 1])
#             res_all[:, -1] = 0
#             res_all[:, 0] = 1
#             if metric == "ibs":
#                 metr_vals.append(metr.ibs_WW(self.y_train, y_tmp, res_all, bins))
#             else:
#                 metr_vals.append(metr.auprc(self.y_train, y_tmp, res_all, bins))
#         plt.plot(range(len(self.models)), metr_vals, label=label)
#
#     def tolerance_find_best(self, ens_metric_name="bic"):
#         pass
