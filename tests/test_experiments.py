import numpy as np
import pandas as pd
import os
import pytest

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.tree import SurvivalTree
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from survivors.tree import CRAID
from survivors.ensemble import BootstrapCRAID
from survivors.ensemble import BoostingCRAID, ProbBoostingCRAID  # SumBoostingCRAID
from survivors.experiments import grid as exp
from survivors import datasets as ds

from survivors.tree.stratified_model import LEAF_MODEL_DICT

from PARAMS.GBSG_PARAM import GBSG_PARAMS
from PARAMS.PBC_PARAM import PBC_PARAMS
from PARAMS.ONK_PARAM import ONK_PARAMS
from PARAMS.WUHAN_PARAM import WUHAN_PARAMS
from PARAMS.COVID_PARAM import COVID_PARAMS

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

SELF_ALGS = {
    "TREE": CRAID,
    "BSTR": BootstrapCRAID,
    "BOOST": BoostingCRAID,
    "PROBOOST": ProbBoostingCRAID,
    # "SUMBOOST": SumBoostingCRAID
}

PARAMS_ = {
    "GBSG": GBSG_PARAMS,
    "PBC": PBC_PARAMS,
    "ONK": ONK_PARAMS,
    "WUHAN": WUHAN_PARAMS,
    "COVID": COVID_PARAMS
}

DATASETS_LOAD = {
    "GBSG": ds.load_gbsg_dataset,
    "PBC": ds.load_pbc_dataset,
    "WUHAN": ds.load_wuhan_dataset,
    "ONK": ds.load_onk_dataset,
    "COVID": ds.load_covid_dataset
}

cox_param_grid = {
    'alpha': [100, 10, 1, 0.1, 0.01, 0.001],
    'ties': ["breslow"]
}
RSF_param_grid = {
    'n_estimators': [30, 50, 100],
    'max_depth': [None, 20],
    'min_samples_leaf': [1, 10, 20],  # [500, 1000, 3000],
    # 'max_features': ["sqrt"],
    "random_state": [123]
}
ST_param_grid = {
    'splitter': ["best", "random"],
    'max_depth': [None, 20, 30],
    'min_samples_leaf': [1, 10, 20],
    'max_features': ["sqrt"],
    "random_state": [123]
}
GBSA_param_grid = {
    'loss': ["coxph"],
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'n_estimators': [30, 50, 100],
    'max_depth': [20, 30],
    'min_samples_leaf': [1, 10, 20],
    'max_features': ["sqrt"],
    "random_state": [123]
}


def get_best_by_full_name(df_full, by_metric="IAUC", choose="max"):
    df = df_full.copy()
    df["METHOD"] = df.apply(lambda x: x["METHOD"].replace("CRAID", f"Tree({x['CRIT']})"), axis=1)
    if not (by_metric in df.columns):
        return None
    best_table = pd.DataFrame([], columns=df.columns)
    for method in df["METHOD"].unique():
        sub_table = df[df["METHOD"] == method]
        if sub_table.shape[0] == 0:
            continue
        if choose == "max":
            best_row = sub_table.loc[sub_table[by_metric].apply(np.mean).idxmax()]
        else:
            best_row = sub_table.loc[sub_table[by_metric].apply(np.mean).idxmin()]
        best_table = best_table.append(dict(best_row), ignore_index=True)
    return best_table


def plot_boxplot_results(df_full, dir_path=None, metrics=[],
                         dataset_name="", all_best=False,
                         by_metric="IAUC", choose="max"):
    if not(all_best):
        df_ = get_best_by_full_name(df_full, by_metric, choose)
    for m in metrics:
        if all_best:
            df_ = get_best_by_full_name(df_full, m, choose="min" if m == "IBS" else "max")
        plt.rcParams.update({'font.size': 15})
        fig, axs = plt.subplots(1, figsize=(8, 8))

        plt.title(f"{dataset_name} {m}")
        plt.boxplot(df_[m][::-1], labels=df_['METHOD'][::-1], showmeans=True, vert=False)
        if dir_path is None:
            plt.show()
        else:
            plt.savefig(os.path.join(dir_path, f"{dataset_name}_{m}_boxplot.png"))
            plt.close(fig)


def import_tables(dirs):
    dfs = []
    for d in dirs:
        dfs.append(pd.read_excel(d))
    df = pd.concat(dfs)
    df = df.drop_duplicates()
    for c in ["IBS", "IAUC", "CI", "CI_CENS"]:
        df[c] = df[c].apply(lambda x: list(map(float, x[1:-1].split())))
    return df


def run(dataset="GBSG", with_self=["TREE", "BSTR", "BOOST"],
        with_external=True, except_stop="all", mode="CV", dir_path=None):
    """
    Conduct experiments for defined dataset and methods (self and external)

    Parameters
    ----------
    dataset : str, optional
        Name of dataset (must be in DATASETS_LOAD). The default is 'GBSG'.
    with_self : list, optional
        Names of self models. The default is ["TREE", "BSTR", "BOOST"].
        "TREE" : CRAID,
        "BSTR" : BootstrapCRAID,
        "BOOST" : BoostingCRAID.
    with_external : boolean, optional
        Flag of addiction external models. The default is True.
        Models : CoxPHSurvivalAnalysis,
                 SurvivalTree,
                 RandomSurvivalForest,
                 GradientBoostingSurvivalAnalysis
    except_stop : str, optional
        Mode of ending because of exception. The default is "all".
        "all" - continue experiments
        else - stop experiments with current method

    dir_path : str, optional
        Path to the final directory

    Returns
    -------
    experim : Experiments
        Instance of class with constructed table.

    Examples
    --------
    from survivors.tests.experiment import run
    df_full, df_best = run()

    """
    if dir_path is None:
        dir_path = os.getcwd() + "\\"
    lst_metrics = ["CI", "CI_CENS", "IBS", "IAUC"]
    if not (dataset in DATASETS_LOAD):
        print("DATASET %s IS NOT DEFINED" % (dataset))
    X, y, features, categ, sch_nan = DATASETS_LOAD[dataset]()
    experim = exp.Experiments(folds=5, except_stop=except_stop, dataset_name=dataset, mode=mode)
    experim.set_metrics(lst_metrics)
    if with_external:
        experim.add_method(CoxPHSurvivalAnalysis, cox_param_grid)
        experim.add_method(SurvivalTree, ST_param_grid)
        experim.add_method(RandomSurvivalForest, RSF_param_grid)
        experim.add_method(GradientBoostingSurvivalAnalysis, GBSA_param_grid)
    if len(with_self) > 0:
        for alg in with_self:
            PARAMS_[dataset][alg]["categ"] = [categ]
            experim.add_method(SELF_ALGS[alg], PARAMS_[dataset][alg])
    experim.run_effective(X, y, dir_path=dir_path, verbose=1)
    return experim


@pytest.fixture(scope="module")
def dir_path():
    return os.path.join(os.getcwd(), "experiment_results")


# @pytest.mark.skip(reason="no way of currently testing this")
@pytest.mark.parametrize(
    "dataset", ["GBSG", "PBC", "WUHAN", "ONK", "COVID"]
    # ["CV", "CV+HOLD-OUT", "TIME-CV"]
)
def test_dataset_exp(dir_path, dataset, mode="CV+HOLD-OUT"):
    prefix = "ABS_BSTR_BALANCE_HONEST"  # "scsurv_models"  # "full_sample_prob"
    # res_exp = run(dataset, with_self=[], with_external=True, mode=mode,  # BOOST
    #               dir_path=dir_path+"\\")  # Only scikit-survival
    res_exp = run(dataset, with_self=["BSTR"], with_external=False, mode=mode,  # BOOST
                  dir_path=dir_path+"\\")  # ["TREE", "BSTR", "BOOST"]BOOST

    df_full = res_exp.get_result()
    df_criterion = res_exp.get_best_by_mode(stratify="balance")  # criterion # get_hold_out_result()
    # df_mode_wei = res_exp.get_best_by_mode(stratify="mode_wei")
    # df_sample = res_exp.get_sample_result()

    df_criterion.to_excel(os.path.join(dir_path, f"{prefix}_{dataset}_{mode}_best.xlsx"), index=False)
    df_full.to_excel(os.path.join(dir_path, f"{prefix}_{dataset}_{mode}_full.xlsx"), index=False)

    # df_mode_wei.to_excel(os.path.join(dir_path, f"{prefix}_strat_mode_wei_{dataset}_{mode}_best.xlsx"), index=False)
    # df_sample.to_excel(os.path.join(dir_path, f"{prefix}_{dataset}_{mode}_sample.xlsx"), index=False)

    # df_best_by_metric_fin = df_best_by_metric.loc[:, ["METHOD", "PARAMS", "CI_mean", "IBS_mean", "IAUC_mean"]].round(5)
    # df_best_by_metric_fin.to_excel(os.path.join(dir_path, f"part_weights_{dataset}_best.xlsx"), index=False)
    # plot_boxplot_results(df_full, dir_path=dir_path,
    #                      metrics=["IBS", "IAUC", "CI"],
    #                      dataset_name=dataset)
