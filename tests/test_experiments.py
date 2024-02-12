import numpy as np
import pandas as pd
import os
import pytest

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.tree import SurvivalTree
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis

from survivors.tree import CRAID
from survivors.ensemble import BootstrapCRAID
from survivors.ensemble import BoostingCRAID, ProbBoostingCRAID
from survivors.ensemble import IBSBoostingCRAID, IBSProbBoostingCRAID  # SumBoostingCRAID
from survivors.ensemble import IBSCleverBoostingCRAID

from survivors.experiments import grid as exp
from survivors import datasets as ds

from survivors.tree.stratified_model import LEAF_MODEL_DICT

from PARAMS.SCHEME_PARAM import SCHEME_PARAMS
from survivors.external import LogLogisticAFT, WeibullAFT, LogNormalAFT
from survivors.external import AFT_param_grid

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

SELF_ALGS = {
    "TREE": CRAID,
    "BSTR": BootstrapCRAID,
    "BOOST": BoostingCRAID,
    "PROBOOST": ProbBoostingCRAID,
    "IBSBOOST": IBSBoostingCRAID,
    "IBSPROBOOST": IBSProbBoostingCRAID,
    "CLEVERBOOST": IBSCleverBoostingCRAID
    # "SUMBOOST": SumBoostingCRAID
}

PARAMS_ = {
    "GBSG": SCHEME_PARAMS,
    "PBC": SCHEME_PARAMS,
    "WUHAN": SCHEME_PARAMS,
    # "ONK": ONK_PARAMS,
    # "COVID": COVID_PARAMS,

    "actg": SCHEME_PARAMS,
    "flchain": SCHEME_PARAMS,
    "smarto": SCHEME_PARAMS,
    "rott2": SCHEME_PARAMS,
    "support2": SCHEME_PARAMS,
    "Framingham": SCHEME_PARAMS,
    "backblaze": SCHEME_PARAMS
}

DATASETS_LOAD = {
    "GBSG": ds.load_gbsg_dataset,
    "PBC": ds.load_pbc_dataset,
    "WUHAN": ds.load_wuhan_dataset,
    # "ONK": ds.load_onk_dataset,
    # "COVID": ds.load_covid_dataset,

    "actg": ds.load_actg_dataset,
    "flchain": ds.load_flchain_dataset,
    "smarto": ds.load_smarto_dataset,
    "rott2": ds.load_rott2_dataset,
    "support2": ds.load_support2_dataset,
    "backblaze": ds.load_backblaze_dataset
    # "Framingham": ds.load_Framingham_dataset
}

cox_param_grid = {
    'alpha': [100, 10, 1, 0.1, 0.01, 0.001],
    'ties': ["breslow"]
}

RSF_param_grid = {
    'n_estimators': [30, 50],  # 100],
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
    'n_estimators': [30, 50],  # 100],
    'max_depth': [20, 30],
    'min_samples_leaf': [1, 10, 20],
    'max_features': ["sqrt"],
    "random_state": [123]
}
CWGBSA_param_grid = {
    'loss': ["coxph"],
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'n_estimators': [30, 50],  # 100],
    'subsample': [0.7, 1.0],
    'dropout_rate': [0.0, 0.1, 0.5],
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
        with_external=True, except_stop="all", mode="CV", dir_path=None, bins_sch="",
        best_metric="IBS", mode_wei=None):
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
    lst_metrics = ["CI", "CI_CENS",
                   "IBS", "BAL_IBS", "IBS_WW", "BAL_IBS_WW", "IBS_REMAIN", "BAL_IBS_REMAIN",
                   "IAUC", "IAUC_WW", "IAUC_TI", "IAUC_WW_TI",
                   "AUPRC", "EVENT_AUPRC", "CENS_AUPRC", "BAL_AUPRC",
                   "KL", "LOGLIKELIHOOD"]
    if not (dataset in DATASETS_LOAD):
        print("DATASET %s IS NOT DEFINED" % (dataset))
    X, y, features, categ, sch_nan = DATASETS_LOAD[dataset]()
    experim = exp.Experiments(folds=5, except_stop=except_stop, dataset_name=dataset, mode=mode, bins_sch=bins_sch)
    experim.set_metrics(lst_metrics)
    experim.add_metric_best(best_metric)
    if with_external:
        experim.add_method(CoxPHSurvivalAnalysis, cox_param_grid)
        experim.add_method(SurvivalTree, ST_param_grid)
        experim.add_method(RandomSurvivalForest, RSF_param_grid)
        experim.add_method(ComponentwiseGradientBoostingSurvivalAnalysis, CWGBSA_param_grid)
        experim.add_method(GradientBoostingSurvivalAnalysis, GBSA_param_grid)
        experim.add_method(LEAF_MODEL_DICT["base"], {})
        experim.add_method(WeibullAFT, AFT_param_grid)
        experim.add_method(LogLogisticAFT, AFT_param_grid)
    if len(with_self) > 0:
        for alg in with_self:
            PARAMS_[dataset][alg]["categ"] = [categ]
            PARAMS_[dataset][alg]["ens_metric_name"] = [best_metric]
            # PARAMS_[dataset][alg]["mode_wei"] = [mode_wei]
            experim.add_method(SELF_ALGS[alg], PARAMS_[dataset][alg])
    experim.run_effective(X, y, dir_path=dir_path, verbose=1)
    return experim


@pytest.fixture(scope="module")
def dir_path():
    return os.path.join(os.getcwd(), "experiment_results", "many_ds")


# @pytest.mark.skip(reason="no way of currently testing this")
# @pytest.mark.parametrize(
#     "bins_sch", ["origin", "rank", "quantile", "log+scale"]
# )
# @pytest.mark.parametrize(
#     "mode_wei", ["exp", "sigmoid", "linear"]  # "exp", "sigmoid" ["likelihood", "conc", "IBS", "IBS_WW", "IBS_REMAIN"]
# )

@pytest.mark.parametrize(
    "best_metric", ["IBS_REMAIN"]  # ["likelihood", "conc", "IBS", "IBS_WW", "IBS_REMAIN"]
)
@pytest.mark.parametrize(
    "dataset",  ["rott2", "PBC", "WUHAN", "GBSG"]  # "flchain", "backblaze", "actg", "support2", "smarto"
)
def test_dataset_exp(dir_path, dataset, best_metric, bins_sch="origin", mode="CV+SAMPLE"):
    mode_wei = None
    # NORMAL_SHORT_QUANTILE_TIME_
    # prefix = f"{best_metric}_STRATTIME+_EXT10_NORMAL_EQ_REG_CLEVERBOOST_SUM_ALL_BINS_{bins_sch}"
    # "scsurv", "bstr_full_WB", SHORT_CNT_DIFF_

    # prefix = f"{best_metric}_STRATTIME+_EXT10_EQ_REG_TREE_ALL_BINS_{bins_sch}"
    prefix = f"{best_metric}_STRATTIME+_EXT10_NORMAL_EQ_REG_TREE_ALL_BINS_{bins_sch}"
    # prefix = f"{best_metric}_STRATTIME+_EXT10_NORMAL_EQ_REG_CLEVERBOOST_ALL_BINS_{bins_sch}"
    # prefix = f"{best_metric}_STRATTIME+_EXT10_NORMAL_EQ_REG_{mode_wei}_PART_BOOST_ALL_BINS_{bins_sch}"

    # prefix = f"{best_metric}_STRATTIME+_scsurv"  # "scsurv", "bstr_full_WB", SHORT_CNT_DIFF_
    # res_exp = run(dataset, with_self=[], with_external=True, mode=mode,
    #               dir_path=dir_path+"\\", bins_sch=bins_sch, best_metric=best_metric)  # Only scikit-survival

    storage_path = os.path.join("D:", os.sep, "Vasilev", "SA", dataset)
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    res_exp = run(dataset, with_self=["TREE"], with_external=False, mode=mode,  # CLEVERBOOST
                  dir_path=storage_path+"\\", bins_sch=bins_sch, best_metric=best_metric, mode_wei=mode_wei)  # ["TREE", "BSTR", "BOOST"]

    df_full = res_exp.get_result()
    df_criterion = res_exp.get_best_by_mode(stratify="criterion")

    df_criterion.to_excel(os.path.join(dir_path, f"{prefix}_{dataset}_{mode}_best.xlsx"), index=False)
    df_full.to_excel(os.path.join(dir_path, f"{prefix}_{dataset}_{mode}_full.xlsx"), index=False)
