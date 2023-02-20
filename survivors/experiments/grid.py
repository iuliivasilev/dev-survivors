import numpy as np
import pandas as pd
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split

from .. import constants as cnt
from .. import metrics as metr
from ..tree.stratified_model import LeafModel


def to_str_from_dict_list(d, strat):
    if isinstance(strat, str):
        return str(d.get(strat, ""))
    elif isinstance(strat, list):
        return ";".join([str(d.get(e, "")) for e in strat])
    return None


def prepare_sample(X, y, train_index, test_index):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    bins = cnt.get_bins(time=y_train[cnt.TIME_NAME], cens=y_train[cnt.CENS_NAME])
    y_train[cnt.TIME_NAME] = np.clip(y_train[cnt.TIME_NAME], bins.min() - 1, bins.max() + 1)
    y_test[cnt.TIME_NAME] = np.clip(y_test[cnt.TIME_NAME], bins.min(), bins.max())
    return X_train, y_train, X_test, y_test, bins


def generate_sample(X, y, folds, mode="CV"):
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
    if mode == "TIME-CV":
        train_index = np.array([], dtype=int)
        for train_index_, test_index_ in skf.split(X, y[cnt.CENS_NAME]):
            if train_index.shape[0] > 0:
                X_train, y_train, X_test, y_test, bins = prepare_sample(X, y, train_index, test_index_)
                yield X_train, y_train, X_test, y_test, bins
            train_index = np.hstack([train_index, test_index_])
    elif mode == "CV+HOLD-OUT":
        X, y, X_HO, y_HO, bins_HO = generate_sample(X, y, folds=1, mode="HOLD-OUT")
        for X_train, y_train, X_test, y_test, bins in generate_sample(X, y, folds=folds, mode="CV"):
            yield X_train, y_train, X_test, y_test, bins
        yield X, y, X_HO, y_HO, bins_HO
    elif mode == "HOLD-OUT":
        for i_fold in range(folds):
            X_TR, X_HO = train_test_split(X, stratify=y[cnt.CENS_NAME],
                                          test_size=0.33, random_state=42 + i_fold)
            X, y, X_HO, y_HO, bins_HO = prepare_sample(X, y, X_TR.index, X_HO.index)
            yield X, y, X_HO, y_HO, bins_HO
    elif mode == "CV":
        for train_index, test_index in skf.split(X, y[cnt.CENS_NAME]):
            X_train, y_train, X_test, y_test, bins = prepare_sample(X, y, train_index, test_index)
            yield X_train, y_train, X_test, y_test, bins
    pass


def count_metric(y_train, y_test, pred_time, pred_surv, pred_haz, bins, metrics_names):
    return np.array([metr.METRIC_DICT[metr_name](y_train, y_test, pred_time, pred_surv, pred_haz, bins)
                     for metr_name in metrics_names])


def crossval_param(method, X, y, folds, metrics_names=['CI'], mode="CV"):
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
        for X_train, y_train, X_test, y_test, bins in generate_sample(X, y, folds, mode):
            est = method(**kwargs)
            if method.__name__.find('CRAID') != -1:  # TODO replace to isinstance
                est.fit(X_train, y_train)
                pred_surv = est.predict_at_times(X_test, bins=bins, mode="surv")
                pred_time = est.predict(X_test, target=cnt.TIME_NAME)
                pred_haz = est.predict_at_times(X_test, bins=bins, mode="hazard")
            elif isinstance(est, LeafModel):
                X_train[cnt.TIME_NAME] = y_train[cnt.TIME_NAME]
                X_train[cnt.CENS_NAME] = y_train[cnt.CENS_NAME]
                est.fit(X_train)
                pred_surv = est.predict_survival_at_times(X_test, bins=bins)
                pred_time = est.predict_feature(X_test, feature_name=cnt.TIME_NAME)
                pred_haz = est.predict_hazard_at_times(X_test, bins=bins)
            else:  # Methods from scikit-survival
                X_train = X_train.fillna(0).replace(np.nan, 0)
                X_test = X_test.fillna(0).replace(np.nan, 0)

                est = est.fit(X_train, y_train)
                survs = est.predict_survival_function(X_test)
                hazards = est.predict_cumulative_hazard_function(X_test)
                pred_surv = np.array(list(map(lambda x: x(bins), survs)))
                pred_haz = np.array(list(map(lambda x: x(bins), hazards)))
                pred_time = -1*est.predict(X_test)
            
            metr_lst.append(count_metric(y_train, y_test, pred_time,
                                         pred_surv, pred_haz, bins, metrics_names))
        return np.vstack(metr_lst)
    return f


class Experiments(object):
    """
    Class receives methods, metrics and grids,
          produces cross-validation experiments,
          stores table of results : name, params, time, metrics (by sample and mean)

    Attributes
    ----------
    methods : list
        Must have predicting methods according to metrics:
            IBS - survival func
            IAUC - cumulative hazard func
            CI - occurred time
            CI_CENS - occurred time
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
    get_agg_results : choose for each method aggregated params by metric and aggreg
    save : export table as xlsx
    
    """
    def __init__(self, folds=5, except_stop="all", dataset_name="NONE_NAME", mode="CV"):
        self.methods = []
        self.methods_grid = []
        self.metrics = ["CI"]
        self.is_table = False
        self.folds = folds
        self.except_stop = except_stop
        self.dataset_name = dataset_name
        self.result_table = None
        self.mode = mode
        
    def add_method(self, method, grid):
        self.methods.append(method)
        self.methods_grid.append(grid)
        
    def set_metrics(self, lst_metric):
        self.metrics = []
        for metr_name in lst_metric:
            if metr_name in metr.METRIC_DICT:
                self.metrics.append(metr_name)
            else:
                print(f"METRIC {metr_name} IS NOT DEFINED")
    
    def run(self, X, y, dir_path=None, verbose=0):
        self.result_table = pd.DataFrame([], columns=["METHOD", "PARAMS", "TIME"] + self.metrics)

        for method, grid in zip(self.methods, self.methods_grid):
            cv_method = crossval_param(method, X, y, self.folds, self.metrics, self.mode)
            print(method, grid)

            grid_params = ParameterGrid(grid)
            p_size = len(grid_params)
            for i_p, p in enumerate(grid_params):
                # try:
                start_time = time.time()
                cv_metr = cv_method(**p)
                full_time = time.time() - start_time
                curr_dict = {"METHOD": method.__name__, "CRIT": p.get("criterion", ""),
                             "PARAMS": str(p), "TIME": full_time}
                cv_metr = {m: cv_metr[:, i] for i, m in enumerate(self.metrics)}
                curr_dict.update(cv_metr)  # dict(zip(self.metrics, cv_metr))
                self.result_table = self.result_table.append(curr_dict, ignore_index=True)
                if verbose > 0:
                    print(f"Iteration: {i_p + 1}/{p_size}")
                    print(f"EXECUTION TIME OF {method.__name__}: {full_time}",
                              {k: [np.mean(v[:-1]), v[-1]] for k, v in cv_metr.items()})  # np.mean(v)
                # except KeyboardInterrupt:
                #     print("HANDELED KeyboardInterrupt")
                #     break
                # except Exception as e:
                #     print("Method: %s, Param: %s finished with except '%s'" % (method.__name__, str(p), e))
                #     if self.except_stop == "all":
                #         break
        if self.mode in ["TIME-CV", "CV+HOLD-OUT"]:
            for m in self.metrics:
                self.result_table[f"{m}_pred_mean"] = self.result_table[m].apply(lambda x: np.mean(x[:-1]))
            for m in self.metrics:
                self.result_table[f"{m}_last"] = self.result_table[m].apply(lambda x: x[-1])

        for m in self.metrics:
            self.result_table[f"{m}_mean"] = self.result_table[m].apply(np.mean)

        self.is_table = True
        if not(dir_path is None):
            # add_time = strftime("%H:%M:%S", gmtime(time.time()))
            self.save(dir_path)

    @staticmethod
    def get_agg_results(result_table, by_metric, choose="median", stratify="criterion"):
        if not (by_metric in result_table.columns):
            return None
        df = result_table.copy()
        stratify_name = f"Stratify({stratify})"
        df[stratify_name] = df["PARAMS"].apply(lambda x: to_str_from_dict_list(eval(x), stratify))
        df["METHOD_FULL"] = df.apply(lambda x: x["METHOD"].replace("CRAID", f"Tree({x[stratify_name]})"), axis=1)

        best_table = pd.DataFrame([], columns=df.columns)
        for method in df['METHOD_FULL'].unique():
            sub_table = df[df["METHOD_FULL"] == method]
            if sub_table.shape[0] == 0:
                continue
            if choose == "max":
                best_row = sub_table.loc[sub_table[by_metric].idxmax()]
            elif choose == "min":
                best_row = sub_table.loc[sub_table[by_metric].idxmin()]
            else:
                best_row = sub_table.sort_values(by=by_metric).iloc[sub_table.shape[0] // 2]
            best_table = best_table.append(dict(best_row), ignore_index=True)
        return best_table

    def get_cv_result(self, stratify="criterion"):
        df_cv_best = self.get_agg_results(self.result_table, "IBS_mean", choose="min", stratify=stratify)
        return df_cv_best

    def get_time_cv_result(self, stratify="criterion"):
        df_time_cv_best = self.get_agg_results(self.result_table, "IBS_pred_mean", choose="min", stratify=stratify)
        return df_time_cv_best

    def get_hold_out_result(self, stratify="criterion"):
        df_hold_out_best = self.get_agg_results(self.result_table, "IBS_pred_mean", choose="min", stratify=stratify)
        rename_d = {metr + "_pred_mean": metr + "_CV_mean" for metr in self.metrics}
        rename_d.update({metr + "_last": metr + "_HO" for metr in self.metrics})
        return df_hold_out_best.rename(rename_d, axis=1)

    def get_result(self):
        return self.result_table

    def get_best_by_mode(self, stratify="criterion"):
        if self.mode == "CV":
            return self.get_cv_result(stratify=stratify)
        elif self.mode == "TIME-CV":
            return self.get_time_cv_result(stratify=stratify)
        elif self.mode == "CV+HOLD-OUT":
            return self.get_hold_out_result(stratify=stratify)
        return None
    
    def save(self, dir_path):
        self.result_table.to_excel(f"{dir_path + self.dataset_name}_FULL_TABLE.xlsx", index=False)
            
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
