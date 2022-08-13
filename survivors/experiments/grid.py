# import sksurv.metrics
# import os
# import random
import numpy as np
import pandas as pd
import time
# from time import strftime, gmtime
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
# from lifelines import KaplanMeierFitter
# from lifelines.utils import concordance_index

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid

from .. import constants as cnt
from .. import metrics as metr


def generate_sample(X, y, folds):
    """
    Generate cross-validate samples with StratifiedKFold.

    Parameters
    ----------
    X : Pandas dataframe
        Contain input features of events.
    y : structured array
        Contain censuring flag and time of events.
    folds : int
        Quantity of cross-validate folds.

    Yields
    ------
    X_train : Pandas dataframe
        Contain input features of train sample.
    y_train : array-like
        Contain censuring flag and time of train sample.
    X_test : Pandas dataframe
        Contain input features of test sample.
    y_test : array-like
        Contain censuring flag and time of test sample.
    bins : array-like
        Points of timeline.

    """
    skf = StratifiedKFold(n_splits=folds)
    for train_index, test_index in skf.split(X, y[cnt.CENS_NAME]):
        X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        bins = cnt.get_bins(time=y_train[cnt.TIME_NAME], cens=y_train[cnt.CENS_NAME])
        y_train[cnt.TIME_NAME] = np.clip(y_train[cnt.TIME_NAME], bins.min()-1, bins.max()+1)
        y_test[cnt.TIME_NAME] = np.clip(y_test[cnt.TIME_NAME], bins.min(), bins.max())
        yield (X_train, y_train, X_test, y_test, bins)
    pass
    

def crossval_param(method, X, y, folds, metrics_names=['CI']):
    """
    Return function, which on sample X, y apply cross-validation and calculate 
    metrics on each folds. 

    Parameters
    ----------
    method : object
        Must have methods for fitting, predicting time, hazard and survival func
            
    X : Pandas dataframe
        Contain input features of events.
    y : structured array
        Contain censuring flag and time of events.
    folds : int
        Quantity of cross-validate folds.
    metrics_names : TYPE, optional
        DESCRIPTION. The default is ['CI'].

    Returns
    -------
    functions
        Recieve hyperparameters and return list of metrics arrays.
        Allow to use in ParameterGrid.

    """
    def f(**kwargs):
        metr_lst = []
        for X_train, y_train, X_test, y_test, bins in generate_sample(X, y, folds):
            if method.__name__.find('CRAID') != -1:  # TODO replace to isinstance
                est = method(**kwargs)
                est.fit(X_train, y_train)
                pred_surv = est.predict_at_times(X_test, bins=bins, mode="surv")
                pred_time = est.predict(X_test, target=cnt.TIME_NAME)
                pred_haz = est.predict_at_times(X_test, bins=bins, mode="hazard")
            else:
                X_train = X_train.fillna(0).replace(np.nan, 0)
                X_test = X_test.fillna(0).replace(np.nan, 0)
                
                est = method(**kwargs).fit(X_train, y_train)
                survs = est.predict_survival_function(X_test)
                hazards = est.predict_cumulative_hazard_function(X_test)
                pred_surv = np.array(list(map(lambda x: x(bins), survs)))
                pred_haz = np.array(list(map(lambda x: x(bins), hazards)))
                pred_time = -1*est.predict(X_test)
            
            metr_lst.append(np.array([metr.METRIC_DICT[metr_name](y_train, y_test, 
                                                        pred_time, pred_surv, pred_haz, bins)
                                      for metr_name in metrics_names]))
        return np.vstack(metr_lst)
    return f


class Experiments(object):
    """
    Class recieves methods, metrics and grids, 
          produces cross-validation experiments,
          stores table of results : name, params, time, metrics (by sample and mean)

    Attributes
    ----------
    methods : list
        Must have predicting methods according to metrics:
            IBS - survival func
            IAUC - cumulative hazard func
            CI - occured time
            CI_CENS - occured time
    methods_grid : list
        Each grid is dictionary: key - param name, values - list
    metrics : list
        Each metric is string, which must be in METRIC_DICT
    is_table : boolean
        Flag of calculation ending
    folds : int
        Quantity of cross-validate folds.
    except_stop : str
        Mode of ending because of exception
        "all" - continue experiments
        else - stop experiments with current method
    dataset_name : str
        Unique name of current dataset (used for saving)

    Methods
    -------
    
    add_method : append method and its grid
    set_metrics : check and set list of metric name 
    run : start experiments with data X, y
    get_best_results : choose for each method best params by metric and aggreg 
    save : export table as xlsx
    
    """
    def __init__(self, folds=5, except_stop="all", dataset_name="NONE_NAME"):
        self.methods = []
        self.methods_grid = []
        self.metrics = ["CI"]
        self.is_table = False
        self.folds = folds
        self.except_stop = except_stop
        self.dataset_name = dataset_name
        self.result_table = None
        
    def add_method(self, method, grid):
        self.methods.append(method)
        self.methods_grid.append(grid)
        
    def set_metrics(self, lst_metric):
        self.metrics = []
        for metr_name in lst_metric:
            if metr_name in metr.METRIC_DICT:
                self.metrics.append(metr_name)
            else:
                print("METRIC %s IS NOT DEFINED" % (metr_name))
    
    def run(self, X, y, dir_path=None, verbose=0):
        self.result_table = pd.DataFrame([], columns=["METHOD", "PARAMS", "TIME"] + self.metrics)
        for method, grid in zip(self.methods, self.methods_grid):
            cv_method = crossval_param(method, X, y, self.folds, self.metrics)
            # try:
            print(method, grid)
            for p in ParameterGrid(grid):
                start_time = time.time()
                cv_metr = cv_method(**p)
                full_time = time.time() - start_time
                curr_dict = {"METHOD": method.__name__, "CRIT": p.get("criterion", ""),
                             "PARAMS": str(p), "TIME": full_time}
                cv_metr = {m: cv_metr[:, i] for i, m in enumerate(self.metrics)}
                curr_dict.update(cv_metr)  # dict(zip(self.metrics, cv_metr))
                self.result_table = self.result_table.append(curr_dict, ignore_index=True)
                if verbose > 0:
                    print("EXECUTION TIME OF %s: %s" % (method.__name__, full_time),
                          {k: np.mean(v) for k, v in cv_metr.items()})
            # except KeyboardInterrupt:
            #     print("HANDELED KeyboardInterrupt")
            #     break
            # except Exception as e:
            #     print("Method: %s, Param: %s finished with except '%s'" % (method.__name__, str(p), e))
            #     if self.except_stop == "all":
            #         break
        for m in self.metrics:
            self.result_table["%s_mean" % (m)] = self.result_table[m].apply(np.mean)
        self.is_table = True
        if not(dir_path is None):
            # add_time = strftime("%H:%M:%S", gmtime(time.time()))
            self.save(dir_path)
    
    def get_result(self):
        return self.result_table
    
    def get_best_results(self, by_metric, choose="max"):
        if not(by_metric in self.metrics):
            return None
        
        best_table = pd.DataFrame([], columns = self.result_table.columns)
        for method in self.result_table['METHOD'].unique():
            sub_table = self.result_table[self.result_table["METHOD"] == method]
            if sub_table.shape[0] == 0:
                continue
            if choose == "max":
                best_row = sub_table.loc[sub_table[by_metric].apply(np.mean).idxmax()]
            else:
                best_row = sub_table.loc[sub_table[by_metric].apply(np.mean).idxmin()]
            best_table = best_table.append(dict(best_row), ignore_index=True)
        return best_table
    
    def save(self, dir_path):
        # save_table = self.result_table.copy()
        # if dir_path.find(".csv") != -1:
        #     save_table.to_csv(dir_path + dataset)
        # else: #to_excel
        self.result_table.to_excel("%s_FULL_TABLE.xlsx" % (dir_path + self.dataset_name), index=False)
            
    # def plot_results(self, dir_path):
    #     df = self.result_table.copy()
    #     df['METHOD'] = df.apply(lambda x: x["METHOD"].replace("CRAID", "Tree(%s)" %(x['CRIT'])),
    #                                  axis = 1)
    #     for m in self.metrics:
    #         fig, axs = plt.subplots(1, figsize=(6, 9))
    #         plt.title("%s %s" % (self.dataset_name, m))
    #         plt.boxplot(df[m], labels = df['METHOD'])
    #         plt.xticks(rotation=90)
    #         plt.savefig(dir_path + self.dataset_name + "%s_boxplot.png" %(m))
    #         plt.close(fig)