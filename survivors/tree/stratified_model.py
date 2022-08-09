import numpy as np
from .. import metrics as metr
from .. import constants as cnt


class LeafModel(object):
    def __init__(self):
        self.shape = None
        self.survival = None
        self.hazard = None
        self.features_mean = dict()

    def fit(self, X_node, need_features=[cnt.TIME_NAME, cnt.CENS_NAME]):
        self.shape = X_node.shape
        self.default_bins = np.array([1, 10, 100, 1000]) #cnt.get_bins(time=X_node[cnt.TIME_NAME].to_numpy(),
                            #             cens=X_node[cnt.CENS_NAME].to_numpy(), mode='a', num_bins=100)
        self.survival = metr.get_survival_func(X_node[cnt.TIME_NAME], X_node[cnt.CENS_NAME])
        self.hazard = metr.get_hazard_func(X_node[cnt.TIME_NAME], X_node[cnt.CENS_NAME])
        self.features_mean = X_node.mean(axis=0).to_dict()
        self.lists = X_node.loc[:, need_features].to_dict(orient="list")

    def get_shape(self):
        return self.shape

    def predict_list_feature(self, feature_name):
        if feature_name in self.lists.keys():
            return self.lists[feature_name]
        return None

    def predict_mean_feature(self, X=None, feature_name=None):
        value = self.features_mean.get(feature_name)
        if X is None:
            return value
        return np.repeat(value, X.shape[0], axis=0)

    def predict_survival_at_times(self, X, bins=None):
        if bins is None:
            bins = self.default_bins
        sf = self.survival.survival_function_at_times(bins).to_numpy()
        return np.repeat(sf[np.newaxis, :], X.shape[0], axis=0)

    def predict_hazard_at_times(self, X, bins=None):
        if bins is None:
            bins = self.default_bins
        hf = self.survival.cumulative_hazard_at_times(bins).to_numpy()
        return np.repeat(hf[np.newaxis, :], X.shape[0], axis=0)
