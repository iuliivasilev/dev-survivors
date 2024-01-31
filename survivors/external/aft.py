from .. import constants as cnt
from ..tree import LeafModel
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter


AFT_param_grid = {
    "penalizer": [0, 0.01, 0.1, 0.5],
    "l1_ratio": [100, 10, 1, 0.1, 0.01, 0.001]
}


class AcceleratedFailureTimeBase(LeafModel):
    base_model = None

    def __init__(self, **kwargs):
        self.model = None
        self.kwargs = kwargs
        super().__init__()

    def fit(self, X_node, need_features=[cnt.TIME_NAME, cnt.CENS_NAME]):
        self.model = self.base_model(**self.kwargs)
        self.model.fit(X_node, cnt.TIME_NAME, event_col=cnt.CENS_NAME)
        super().fit(X_node, need_features)

    def predict_survival_at_times(self, X=None, bins=None):
        return self.model.predict_survival_function(X, times=bins).to_numpy().T

    def predict_hazard_at_times(self, X=None, bins=None):
        return self.model.predict_hazard(X, times=bins).to_numpy().T

    def predict_feature(self, X=None, feature_name=None):
        if feature_name == cnt.TIME_NAME:
            return self.model.predict_expectation(X).to_numpy()
        return super().predict_feature(X, feature_name)


class WeibullAFT(AcceleratedFailureTimeBase):
    base_model = WeibullAFTFitter


class LogNormalAFT(AcceleratedFailureTimeBase):
    base_model = LogNormalAFTFitter


class LogLogisticAFT(AcceleratedFailureTimeBase):
    base_model = LogLogisticAFTFitter
