import numpy as np

from ..tree import CRAID
from .. import constants as cnt
from .base_ensemble import FastBaseEnsemble
from joblib import Parallel, delayed


class BootstrapCRAID(FastBaseEnsemble):
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
            X_sub_tr, y_sub_tr = cnt.pd_to_xy(x_sub)

            model = CRAID(features=self.features, random_state=i, **self.tree_kwargs)
            model.fit(X_sub_tr, y_sub_tr)

            self.add_model(model, x_oob)
        print(f"fitted: {len(self.models)} models.")


class ParallelBootstrapCRAID(BootstrapCRAID):

    @staticmethod
    def fit_tree(x_sub, params):
        X_tr, y_tr = cnt.pd_to_xy(x_sub)
        model = CRAID(**params)
        model.fit(X_tr, y_tr)
        return model

    def fit(self, X, y):
        self.features = X.columns
        X = X.reset_index(drop=True)
        X[cnt.CENS_NAME] = y[cnt.CENS_NAME].astype(np.int32)
        X[cnt.TIME_NAME] = y[cnt.TIME_NAME].astype(np.float32)

        self.X_train = X
        self.y_train = y
        self.update_params()

        p_s = []
        x_sub_ind = []
        for i in range(self.n_estimators):
            x_sub = self.X_train.sample(n=self.size_sample, replace=self.bootstrap, random_state=i)
            x_sub_ind.append(x_sub.index)

            x_sub = x_sub.reset_index(drop=True)
            params = self.tree_kwargs.copy()
            params['random_state'] = i
            params['features'] = self.features
            p_s.append({"x_sub": x_sub, "params": params})

        with Parallel(n_jobs=self.tree_kwargs.get("n_jobs", 10), verbose=False, batch_size=10) as parallel:
            ml = parallel(delayed(self.fit_tree)(**p) for p in p_s)

        for model, x_sub_ind in zip(ml, x_sub_ind):
            x_oob = self.X_train.loc[self.X_train.index.difference(x_sub_ind), :]
            self.add_model(model, x_oob)
        # print(f"fitted: {len(self.models)} models.")

    # def predict_at_times(self, x_test, bins, aggreg=True, mode="surv"):
    #     with Parallel(n_jobs=self.tree_kwargs.get("n_jobs", 10), verbose=False, batch_size=10) as parallel:
    #         ml = parallel(delayed(lambda m: m.predict_at_times(x_test, bins=bins,
    #                                                            mode=mode)[np.newaxis, :])(m) for m in self.models)
    #     res = np.vstack(ml)
    #     if aggreg:
    #         res = self.get_aggreg(res)
    #         if mode == "surv":
    #             res[:, -1] = 0
    #             res[:, 0] = 1
    #     return res
    #
    # def predict(self, x_test, aggreg=True, **kwargs):
    #     with Parallel(n_jobs=self.tree_kwargs.get("n_jobs", 10), verbose=False, batch_size=10) as parallel:
    #         ml = parallel(delayed(lambda m: m.predict(x_test, **kwargs))(m) for m in self.models)
    #     res = np.array(ml)
    #     if aggreg:
    #         res = self.get_aggreg(res)
    #     return res
