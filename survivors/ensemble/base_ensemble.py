import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

from .. import metrics as metr
from .. import constants as cnt


class BaseEnsemble(object):
    """
    Base ensemble of survival decision tree.

    Attributes
    ----------
    size_sample : float
        Size of generated subsample
    n_estimators : int
        Number of base models
    ens_metric_name : str
        Metric defines quantitative of ensemble
    descend_metr : boolean
        Flag of descend for ens_metric_name
    bootstrap : boolean
        Flag for using bootstrap sampling (with return)
    tolerance : boolean
        Flag for fitting full ensemble and choosing best submodels
    aggreg : boolean
        Flag of aggregating base responses
    aggreg_func : str
        Function of aggregating (if aggreg)
    tree_kwargs : dict
        Parameters for building base models

    models : list
        Base models of ensemble (for example, CRAID)
    oob : list
        Out of bag sample for each model
    ens_metr : array-like
        Values of ens_metric_name for ensemble with i models
    list_pred_oob : list
        Predictions of ensemble models[:i+1]

    Methods
    -------
    update_params : attributes preparation
    fit : build ensemble with X, y data (abstract)
    add_model : updating ensemble with new model and oob
    select_model : remaining fixed models
    tolerance_find_best : iterative method of selecting best sub ensemble

    predict : return values of features, rules or schemes (look at CRAID)
    predict_at_times : return survival or hazard function

    score_oob : calculate metric "ens_metric_name" for ensemble
    """

    def __init__(self, size_sample=0.7,
                 n_estimators=10,
                 aggreg=True,
                 ens_metric_name="roc",
                 bootstrap=True,
                 tolerance=True,
                 aggreg_func="mean",
                 **tree_kwargs):
        self.size_sample = size_sample
        self.n_estimators = n_estimators
        self.aggreg = aggreg
        # self.ens_metric_name = ens_metric_name
        # self.descend_metr = self.ens_metric_name in metr.DESCEND_METRICS
        self.bootstrap = bootstrap
        self.tolerance = tolerance
        self.tree_kwargs = tree_kwargs
        self.aggreg_func = aggreg_func
        self.X_train = None
        self.y_train = None
        self.features = None

    def update_params(self):
        self.models = []
        self.oob = []
        self.ens_metr = np.zeros(self.n_estimators)
        self.list_pred_oob = []

        if isinstance(self.size_sample, float):
            self.size_sample = int(self.size_sample * self.X_train.shape[0])
        self.bins = cnt.get_bins(time=self.y_train[cnt.TIME_NAME],
                                 cens=self.y_train[cnt.CENS_NAME])

        cnt.set_seed(10)

    def fit(self):
        pass

    def add_model(self, model, x_oob):
        self.models.append(model)
        self.oob.append(x_oob)
        if self.ens_metric_name == "conc":
            tree_pred = pd.DataFrame(model.predict(x_oob, target=cnt.TIME_NAME), index=x_oob.index)
        elif self.ens_metric_name == "roc":
            tree_pred = pd.DataFrame(model.predict(x_oob, target=cnt.CENS_NAME), index=x_oob.index)
        else:
            tree_pred = pd.DataFrame(model.predict_at_times(x_oob, bins=self.bins, mode="surv"), index=x_oob.index)
            tree_pred = tree_pred.apply(lambda r: np.array(r), axis=1)
        self.list_pred_oob.append(tree_pred)

    def select_model(self, start, end):
        self.models = self.models[start:end]
        self.oob = self.oob[start:end]
        self.list_pred_oob = self.list_pred_oob[start:end]

    def tolerance_find_best(self):
        if self.descend_metr:
            max_index = np.argmin(self.ens_metr)
        else:
            max_index = np.argmax(self.ens_metr)
        self.select_model(0, max_index + 1)
        print(self.ens_metr)

    def get_aggreg(self, x):
        if self.aggreg_func == "median":
            return np.median(x, axis=0)
        elif self.aggreg_func == "mean":
            return np.mean(x, axis=0)
        return None

    def predict_at_times(self, x_test, bins, aggreg=True, mode="surv"):
        res = []
        for i in range(len(self.models)):
            res.append(self.models[i].predict_at_times(x_test, bins=bins,
                                                       mode=mode)[np.newaxis, :])
        res = np.vstack(res)
        if aggreg:
            res = self.get_aggreg(res)
            if mode == "surv":
                res[:, -1] = 0
                res[:, 0] = 1
        return res

    def predict(self, x_test, aggreg=True, **kwargs):
        res = []
        for i in range(len(self.models)):
            res.append(self.models[i].predict(x_test, **kwargs))
        res = np.array(res)
        if aggreg:
            res = self.get_aggreg(res)
        return res

    """ SCORES """

    def score_oob(self):
        if self.aggreg:
            score = self.aggregate_score_selfoob()
        else:
            score = self.separate_score_oob()
        return np.round(score, 4)

    def separate_score_oob(self):
        scores = np.array([])
        for i in range(len(self.models)):
            X_v = self.oob[i]
            if self.ens_metric_name == "conc":
                pred = self.models[i].predict(X_v, target=cnt.TIME_NAME)
                score = concordance_index(X_v[cnt.TIME_NAME], pred)
            else:
                pred = self.models[i].predict_at_times(X_v, bins=self.bins, mode="surv")
                y_true = cnt.get_y(X_v[cnt.CENS_NAME], X_v[cnt.TIME_NAME])
                score = metr.ibs(self.y_train, y_true, pred, self.bins)
            scores = np.append(scores, score)
        return np.round(np.mean(scores), 4)

    def aggregate_score_selfoob(self, bins=None):
        is_ibs = self.ens_metric_name.upper().find("IBS") >= 0
        if self.ens_metric_name in ["conc"] or is_ibs:
            list_target_time = [oob_[cnt.TIME_NAME].to_frame() for oob_ in self.oob]
            target_time = pd.concat(list_target_time, axis=1).mean(axis=1)

        if self.ens_metric_name in ["roc"] or is_ibs:
            list_target_cens = [oob_[cnt.CENS_NAME].to_frame() for oob_ in self.oob]
            target_cens = pd.concat(list_target_cens, axis=1).mean(axis=1)

        if self.ens_metric_name in ["conc", "roc"]:
            pred = pd.concat(self.list_pred_oob, axis=1).mean(axis=1)
            if self.ens_metric_name == "conc":
                return concordance_index(target_time, pred)
            return roc_auc_score(target_cens, pred)

        if is_ibs:
            pred = pd.concat(self.list_pred_oob, axis=1).apply(lambda r: r.mean(axis=0), axis=1)
            pred = np.array(pred.to_list())
            y_true = cnt.get_y(target_cens, target_time)
            return metr.ibs(self.y_train, y_true, pred, self.bins)
        return None


class FastBaseEnsemble(BaseEnsemble):
    def update_params(self):
        self.models = []
        self.oob_index = []

        if isinstance(self.size_sample, float):
            self.size_sample = int(self.size_sample * self.X_train.shape[0])
        self.bins = cnt.get_bins(time=self.y_train[cnt.TIME_NAME],
                                 cens=self.y_train[cnt.CENS_NAME])
        cnt.set_seed(10)

    def add_model(self, model, x_oob):
        self.models.append(model)
        self.oob_index.append(x_oob.index.values)

    def select_model(self, start, end):
        self.models = self.models[start:end]
        self.oob_index = self.oob_index[start:end]

    def tolerance_find_best(self, ens_metric_name="bic"):
        self.ens_metric_name = ens_metric_name
        ens_metr_arr = np.zeros(self.n_estimators)

        self.prepare_for_tolerance()
        for i in range(self.n_estimators):
            self.tolerance_iter = i
            self.predict_by_i(i)
            ens_metr_arr[i] = self.score_oob()

        descend_metr = self.ens_metric_name in metr.DESCEND_METRICS
        if descend_metr:
            best_index = np.argmin(ens_metr_arr)
        else:
            best_index = np.argmax(ens_metr_arr)
        self.select_model(0, best_index + 1)
        print(ens_metr_arr)
        print(f"fitted: {len(self.models)} models.")

    def prepare_for_tolerance(self):
        if self.ens_metric_name in ["iauc", "likelihood", "bic"] or self.ens_metric_name.upper().find("IBS") >= 0:
            dim = (self.X_train.shape[0], self.bins.shape[0])
        else:
            dim = (self.X_train.shape[0])
        self.oob_prediction = np.zeros(dim, dtype=np.float)

        if self.ens_metric_name in ["likelihood", "bic"]:
            self.oob_prediction_hf = np.zeros(dim, dtype=np.float)
        self.oob_count = np.zeros((self.X_train.shape[0]), dtype=np.int)

    def predict_by_i(self, ind_model):
        model = self.models[ind_model]
        oob_index = self.oob_index[ind_model]
        x_oob = self.X_train.iloc[oob_index, :]

        self.oob_count[oob_index] += 1
        if self.ens_metric_name == "conc":
            self.oob_prediction[oob_index] += model.predict(x_oob, target=cnt.TIME_NAME)
        elif self.ens_metric_name == "roc":
            self.oob_prediction[oob_index] += model.predict(x_oob, target=cnt.CENS_NAME)
        elif self.ens_metric_name in ["likelihood", "bic"]:
            self.oob_count = np.ones((self.X_train.shape[0]), dtype=np.int)
            self.oob_prediction_hf += model.predict_at_times(self.X_train, bins=self.bins, mode="hazard")
            self.oob_prediction += model.predict_at_times(self.X_train, bins=self.bins, mode="surv")
        else:
            self.oob_prediction[oob_index] += model.predict_at_times(x_oob, bins=self.bins, mode="surv")

    def aggregate_score_selfoob(self):
        index_join_oob = np.where(self.oob_count != 0)
        is_ibs = self.ens_metric_name.upper().find("IBS") >= 0
        if is_ibs:
            pred = self.oob_prediction[index_join_oob] / self.oob_count[index_join_oob][:, None]
        elif self.ens_metric_name in ["likelihood", "bic"]:
            pred_hf = self.oob_prediction_hf[index_join_oob]  # / self.oob_count[index_join_oob][:, None]
            pred_sf = self.oob_prediction[index_join_oob]  # / self.oob_count[index_join_oob][:, None]
        else:
            pred = self.oob_prediction[index_join_oob] / self.oob_count[index_join_oob]

        join_oob = self.y_train[index_join_oob]
        target_time = join_oob[cnt.TIME_NAME]
        target_cens = join_oob[cnt.CENS_NAME]

        if self.ens_metric_name == "conc":
            return concordance_index(target_time, pred)
        elif self.ens_metric_name == "roc":
            return roc_auc_score(target_cens, pred)
        elif is_ibs:
            y_true = cnt.get_y(target_cens, target_time)
            return metr.METRIC_DICT[self.ens_metric_name.upper()](self.y_train, y_true, None, pred, None, self.bins)
        elif self.ens_metric_name == "likelihood":
            return metr.loglikelihood(target_time, target_cens, pred_sf, pred_hf, self.bins)
        elif self.ens_metric_name == "bic":
            return metr.bic(self.tolerance_iter+1, self.size_sample, target_time, target_cens, pred_sf, pred_hf, self.bins)
        return None
