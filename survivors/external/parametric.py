from .. import constants as cnt
from .leaf_model import LeafModel
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
from lifelines import CoxPHFitter
import numpy as np


PARAM_GRID = {
    "penalizer": [0, 0.01, 0.1, 0.5, 1.0],
    "l1_ratio": [100, 10, 1, 0.1, 0.01, 0.001]
}

AFT_param_grid = PARAM_GRID.copy()
CoxPH_param_grid = PARAM_GRID.copy()


class ParametricLifelinesBase(LeafModel):
    base_model = None

    def __init__(self, except_stop=True, **kwargs):
        self.model = None
        self.except_stop = except_stop
        super().__init__(**kwargs)

    def prepare_data(self, X):
        if X is None:
            if self.model is None:
                raise Exception("Model does not exist")
            return self.model._central_values

        if self.model is None:  # Prepare train sample
            self.features = set(self.features) & set((X.notna().sum() != 0).index)  # Delete full nan
            self.features = set(self.features) & set((X.std() == 0).index)  # Delete columns with one value
            self.features = list(self.features | {cnt.TIME_NAME, cnt.CENS_NAME})
            if X[cnt.CENS_NAME].sum() == 0:
                X[cnt.CENS_NAME] = 1  # parametric models do not work with only censored observations
        X = X[list(set(self.features) & set(X.columns))].copy()
        X = X.fillna(0).replace(np.nan, 0)
        return X

    def fit(self, X_node):
        super().fit(X_node)
        X_node = self.prepare_data(X_node)
        self.model = self.base_model(**self.kwargs)
        try:
            self.model.fit(X_node, cnt.TIME_NAME, event_col=cnt.CENS_NAME)
        except Exception as e:
            if self.except_stop:
                raise Exception(f"Leaf model fitting raise exception: {str(e)}")
            self.features = [cnt.TIME_NAME, cnt.CENS_NAME]
            self.model.fit(X_node[self.features], cnt.TIME_NAME, event_col=cnt.CENS_NAME)

    def predict_survival_at_times(self, X=None, bins=None):
        X = self.prepare_data(X)
        return self.model.predict_survival_function(X, times=bins).to_numpy().T

    def predict_hazard_at_times(self, X=None, bins=None):
        X = self.prepare_data(X)
        chf = self.model.predict_cumulative_hazard(X, times=bins).to_numpy().T
        return chf

    def predict_feature(self, X=None, feature_name=None):
        X = self.prepare_data(X)
        if feature_name == cnt.TIME_NAME:
            return self.model.predict_expectation(X).to_numpy()
        return super().predict_feature(X, feature_name)


class WeibullAFT(ParametricLifelinesBase):
    base_model = WeibullAFTFitter


class LogNormalAFT(ParametricLifelinesBase):
    base_model = LogNormalAFTFitter


class LogLogisticAFT(ParametricLifelinesBase):
    base_model = LogLogisticAFTFitter


class CoxPH(ParametricLifelinesBase):
    base_model = CoxPHFitter


LEAF_AFT_DICT = {
    "WeibullAFT": WeibullAFT,
    "LogNormalAFT": LogNormalAFT,
    "LogLogisticAFT": LogLogisticAFT,
    "CoxPH": CoxPH
}
