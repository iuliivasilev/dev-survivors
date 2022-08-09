import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

from .. import metrics as metr
from ..tree import CRAID
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
        Metric defines quantitve of ensemble
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
    tolerance_find_best : iterative method of selecting best subensemble
    
    predict : return values of features, rules or schemes (look at CRAID)
    predict_at_times : return survival or hazard function

    score_oob : calculate metric "ens_metric_name" for ensemble
    
    """
    def __init__(self, size_sample = 0.7,
                 n_estimators = 10,
                 aggreg = True,
                 ens_metric_name = "roc",
                 bootstrap = True,
                 tolerance = True,
                 aggreg_func = "mean",
                 **tree_kwargs):
        self.size_sample = size_sample
        self.n_estimators = n_estimators
        self.aggreg = aggreg
        self.ens_metric_name = ens_metric_name
        self.descend_metr = self.ens_metric_name in metr.DESCEND_METRICS
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
        
        if self.size_sample < 1.0:
            self.size_sample = int(self.size_sample*self.X_train.shape[0])
        self.bins = cnt.get_bins(time = self.y_train[cnt.TIME_NAME], 
                                 cens = self.y_train[cnt.CENS_NAME])
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
        self.select_model(0,max_index+1)
        print(self.ens_metr)
    
    def get_aggreg(self, x):
        if self.aggreg_func == 'median':
            return np.median(x, axis = 0)
        elif self.aggreg_func == "mean":
            return np.mean(x, axis = 0)
        return None
    
    def predict_at_times(self, x_test, bins, aggreg = True, mode = "surv"):
        res = []
        for i in range(len(self.models)):
            res.append(self.models[i].predict_at_times(x_test, bins = bins, 
                                                       mode = mode)[np.newaxis, :])
        res = np.vstack(res)
        if aggreg:
            res = self.get_aggreg(res)
        return res
    
    def predict(self, x_test, aggreg = True, **kwargs):
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
                pred = self.models[i].predict(X_v, target = cnt.TIME_NAME)
                score = concordance_index(X_v[cnt.TIME_NAME], pred)
            else:
                pred = self.models[i].predict_at_times(X_v, bins = self.bins, mode = "surv")
                y_true = cnt.get_y(X_v[cnt.CENS_NAME], X_v[cnt.TIME_NAME])
                score = metr.ibs(self.y_train, y_true, pred, self.bins)
            scores = np.append(scores, score)
        return np.round(np.mean(scores), 4)
        
    def aggregate_score_selfoob(self, bins = None):
        if self.ens_metric_name in ["conc", "ibs"]:
            list_target_time = [oob_[cnt.TIME_NAME].to_frame() for oob_ in self.oob]
            target_time = pd.concat(list_target_time, axis=1).mean(axis = 1)
            
        if self.ens_metric_name in ["roc", "ibs"]:
            list_target_cens = [oob_[cnt.CENS_NAME].to_frame() for oob_ in self.oob]
            target_cens = pd.concat(list_target_cens, axis=1).mean(axis = 1)
            
        if self.ens_metric_name in ["conc", "roc"]:
            pred = pd.concat(self.list_pred_oob, axis=1).mean(axis = 1)
            if self.ens_metric_name == "conc":
                return concordance_index(target_time, pred)
            return roc_auc_score(target_cens, pred)
        
        if self.ens_metric_name == "ibs":
            pred = pd.concat(self.list_pred_oob, axis=1).apply(lambda r: r.mean(axis = 0),axis = 1)
            pred = np.array(pred.to_list())
            y_true = cnt.get_y(target_cens, target_time)
            return metr.ibs(self.y_train, y_true, pred, self.bins)
        return None