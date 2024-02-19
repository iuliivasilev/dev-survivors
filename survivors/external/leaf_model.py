import numpy as np
from .. import constants as cnt


class LeafModel(object):

    def __init__(self, weights_name=None):
        self.shape = None
        self.features_predict = dict()
        self.lists = dict()
        self.default_bins = np.array([1, 10, 100, 1000])
        self.weights_name = weights_name

    def fit(self, X_node, need_features=[cnt.TIME_NAME, cnt.CENS_NAME], *args, **kwargs):
        X_sub = X_node[need_features]
        self.shape = X_sub.shape
        self.features_predict = X_sub.mean(axis=0).to_dict()
        self.lists = X_sub.to_dict(orient="list")
        if self.weights_name is None:
            self.weights = None
        else:
            if self.weights_name in X_node.columns:
                self.weights = X_node[self.weights_name].to_numpy()
            else:
                self.weights = np.ones(self.shape[0])

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


class NonparamLeafModel(LeafModel):
    survival_class = None
    hazard_class = None

    def __init__(self, *args, **kwargs):
        if (self.survival_class is None) and (self.hazard_class is None):
            raise Exception("There is no survival or hazard base class!")
        self.survival = None
        self.hazard = None
        super().__init__(*args, **kwargs)

    def predict_survival_at_times(self, X=None, bins=None):
        if self.survival_class is None:
            hf = self.predict_hazard_at_times(X=X, bins=bins)
            sf = np.exp(-1 * hf)
            return sf

        if self.survival is None:
            self.survival = self.survival_class()
            self.survival.fit(self.lists[cnt.TIME_NAME],
                              self.lists[cnt.CENS_NAME],
                              self.weights)
        if bins is None:
            bins = self.survival.timeline
        sf = self.survival.survival_function_at_times(bins)
        if X is None:
            return sf
        return np.repeat(sf[np.newaxis, :], X.shape[0], axis=0)

    def predict_hazard_at_times(self, X=None, bins=None):
        if self.hazard_class is None:
            sf = self.predict_survival_at_times(X=X, bins=bins)
            hf = -1 * np.log(sf + 1e-100)
            return hf

        if self.hazard is None:
            self.hazard = self.hazard_class()
            self.hazard.fit(self.lists[cnt.TIME_NAME],
                            self.lists[cnt.CENS_NAME],
                            self.weights)
        if bins is None:
            bins = self.hazard.timeline
        hf = self.hazard.cumulative_hazard_at_times(bins)
        if X is None:
            return hf
        return np.repeat(hf[np.newaxis, :], X.shape[0], axis=0)


class NormalizedLeafModel(NonparamLeafModel):
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)

        durs = np.random.normal(np.mean(self.lists[cnt.TIME_NAME]),
                                np.std(self.lists[cnt.TIME_NAME]) / np.sqrt(2), 1000)
        events = np.random.choice(self.lists[cnt.CENS_NAME], size=1000, replace=True)
        self.lists[cnt.CENS_NAME] = events[durs > 0]
        self.lists[cnt.TIME_NAME] = durs[durs > 0]
        self.weights = np.ones_like(self.lists[cnt.TIME_NAME])


class MeaningLeafModel(NonparamLeafModel):
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self.lists[cnt.TIME_NAME] = np.random.choice(self.lists[cnt.TIME_NAME], size=(2, 1000), replace=True).mean(axis=0)
        self.lists[cnt.CENS_NAME] = np.random.choice(self.lists[cnt.CENS_NAME], size=1000, replace=True)
        self.weights = np.ones_like(self.lists[cnt.TIME_NAME])


class MixLeafModel(NormalizedLeafModel):
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self.lists[cnt.TIME_NAME] = np.random.choice(self.lists[cnt.TIME_NAME], size=(2, 1000), replace=True).mean(axis=0)
        self.lists[cnt.CENS_NAME] = np.random.choice(self.lists[cnt.CENS_NAME], size=1000, replace=True)
        self.weights = np.ones_like(self.lists[cnt.TIME_NAME])
