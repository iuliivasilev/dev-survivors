import numpy as np
from .. import metrics as metr
from .. import constants as cnt


class LeafModel(object):
    def __init__(self):
        self.shape = None
        self.survival = None
        self.hazard = None
        self.features_predict = dict()
        self.lists = dict()
        self.default_bins = np.array([1, 10, 100, 1000])

    def fit(self, X_node, need_features=[cnt.TIME_NAME, cnt.CENS_NAME]):
        self.shape = X_node.shape
        self.features_predict = X_node.mean(axis=0).to_dict()
        self.lists = X_node.loc[:, need_features].to_dict(orient="list")

    def get_shape(self):
        return self.shape

    def predict_list_feature(self, feature_name):
        if feature_name in self.lists.keys():
            return np.array(self.lists[feature_name])
        return None

    def predict_feature(self, X=None, feature_name=None):
        value = self.features_predict.get(feature_name)
        if X is None:
            return value
        return np.repeat(value, X.shape[0], axis=0)

    def predict_survival_at_times(self, X=None, bins=None):
        pass

    def predict_hazard_at_times(self, X=None, bins=None):
        pass


class LeafSurviveAndHazard(LeafModel):
    def __init__(self):
        super().__init__()

    def predict_survival_at_times(self, X=None, bins=None):
        if self.survival is None:
            self.survival = metr.get_survival_func(self.lists[cnt.TIME_NAME], self.lists[cnt.CENS_NAME])
        if bins is None:
            bins = self.default_bins
        sf = self.survival.survival_function_at_times(bins).to_numpy()
        if X is None:
            return sf
        return np.repeat(sf[np.newaxis, :], X.shape[0], axis=0)

    def predict_hazard_at_times(self, X=None, bins=None):
        if self.hazard is None:
            self.hazard = metr.get_hazard_func(self.lists[cnt.TIME_NAME], self.lists[cnt.CENS_NAME])
        if bins is None:
            bins = self.default_bins
        hf = self.hazard.cumulative_hazard_at_times(bins).to_numpy()
        if X is None:
            return hf
        return np.repeat(hf[np.newaxis, :], X.shape[0], axis=0)


class LeafOnlyHazardModel(LeafSurviveAndHazard):
    def predict_survival_at_times(self, X=None, bins=None):
        hf = self.predict_hazard_at_times(X=X, bins=bins)
        sf = np.exp(-1*hf)
        if X is None:
            return sf
        return sf


class LeafOnlySurviveModel(LeafSurviveAndHazard):
    def predict_hazard_at_times(self, X=None, bins=None):
        sf = self.predict_survival_at_times(X=X, bins=bins)
        hf = -1*np.log(sf)
        if X is None:
            return hf
        return hf


class KaplanMeier:
    def __init__(self):
        self.timeline = None
        self.survival_function = None

    def fit(self, durations, right_censor, weights):
        self.timeline = np.unique(durations)

        dur_ = np.searchsorted(self.timeline, durations)
        hist_dur = np.bincount(dur_, weights=weights)
        hist_cens = np.bincount(dur_, weights=right_censor*weights)
        cumul_hist = np.cumsum(hist_dur[::-1])[::-1]
        self.survival_function = np.hstack([1.0, np.cumprod((1.0 - hist_cens / (cumul_hist)))])

    def survival_function_at_times(self, times):
        place_bin = np.digitize(times, self.timeline)
        return self.survival_function[np.clip(place_bin, 0, None)]


class NelsonAalen:
    def __init__(self, smoothing=True):
        self.timeline = None
        self.survival_function = None
        self.smoothing = smoothing

    def fit(self, durations, right_censor, weights):
        self.timeline = np.unique(durations)

        dur_ = np.searchsorted(self.timeline, durations)
        hist_dur = np.bincount(dur_, weights=weights)
        hist_cens = np.bincount(dur_, weights=right_censor*weights)
        cumul_hist_dur = np.cumsum(hist_dur[::-1])[::-1]
        if self.smoothing and all(weights == 1):
            cumul_hist_dur = cumul_hist_dur.astype("int")
            hist_cens = hist_cens.astype("int")
            cum_ = np.cumsum(1.0 / np.arange(1, np.max(cumul_hist_dur) + 1))
            hf = cum_[cumul_hist_dur - 1] - np.where(cumul_hist_dur - hist_cens - 1 >= 0,
                                                     cum_[cumul_hist_dur - hist_cens - 1], 0)
        else:
            hf = hist_cens / cumul_hist_dur
        self.hazard_function = np.hstack([0.0, np.cumsum(hf)])

    def cumulative_hazard_at_times(self, times):
        place_bin = np.digitize(times, self.timeline)
        return self.hazard_function[np.clip(place_bin, 0, None)]


class WeightSurviveModel(LeafModel):
    def __init__(self, weights_name="weights_obs"):
        self.shape = None
        self.survival = None
        self.hazard = None
        self.features_predict = dict()
        self.lists = dict()
        self.weights_name = weights_name
        self.default_bins = np.array([1, 10, 100, 1000])

    def fit(self, X_node, need_features=[cnt.TIME_NAME, cnt.CENS_NAME]):
        if self.weights_name is None:
            self.weights = np.ones_like(X_node[cnt.TIME_NAME])
        else:
            self.weights = X_node[self.weights_name].to_numpy()
        self.survival = KaplanMeier()
        self.survival.fit(X_node[cnt.TIME_NAME].to_numpy(),
                          X_node[cnt.CENS_NAME].to_numpy(),
                          self.weights)
        self.hazard = NelsonAalen()
        self.hazard.fit(X_node[cnt.TIME_NAME].to_numpy(),
                        X_node[cnt.CENS_NAME].to_numpy(),
                        self.weights)
        super().fit(X_node, need_features)

    def predict_survival_at_times(self, X=None, bins=None):
        if bins is None:
            bins = self.default_bins
        sf = self.survival.survival_function_at_times(bins)
        if X is None:
            return sf
        return np.repeat(sf[np.newaxis, :], X.shape[0], axis=0)

    def predict_hazard_at_times(self, X=None, bins=None):
        if bins is None:
            bins = self.default_bins
        hf = self.hazard.cumulative_hazard_at_times(bins)
        if X is None:
            return hf
        return np.repeat(hf[np.newaxis, :], X.shape[0], axis=0)


class BaseFastSurviveModel(WeightSurviveModel):
    def __init__(self):
        super().__init__(weights_name=None)


LEAF_MODEL_DICT = {
    "base": LeafSurviveAndHazard,
    "only_hazard": LeafOnlyHazardModel,
    "only_survive": LeafOnlySurviveModel,
    "wei_survive": WeightSurviveModel,
    "base_fast": BaseFastSurviveModel
}
