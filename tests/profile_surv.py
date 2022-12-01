import cProfile
import pstats

import survivors.datasets as ds
from survivors.experiments.grid import generate_sample

from test_experiments import run

from survivors.constants import get_bins, TIME_NAME, CENS_NAME
from survivors.tree import CRAID
from survivors.ensemble import BoostingCRAID

DATASETS_LOAD = {
    "GBSG": ds.load_gbsg_dataset,
    "PBC": ds.load_pbc_dataset,
    "WUHAN": ds.load_wuhan_dataset,
    "ONK": ds.load_onk_dataset,
    "COVID": ds.load_covid_dataset
}

TREE_PARAMS = {
    # Baseline time: 83,97 -> 7.11
    "PBC": {'categ': ['trt', 'sex', 'ascites', 'hepato', 'spiders'],
            'criterion': 'peto', 'cut': True, 'depth': 10,
            'max_features': 1.0, 'min_samples_leaf': 5, 'signif': 0.05, 'woe': False},
    # Baseline time: 108,19 -> 10.4
    "ONK": {'categ': ['Диагноз'], 'criterion': 'peto', 'cut': True,
            'depth': 10, 'max_features': 1.0, 'min_samples_leaf': 100,
            'signif': 0.05, 'woe': False}
}

BOOST_PARAMS = {
    # Baseline time: 418,59 -> 86.3
    "PBC": {'aggreg_func': 'wei', 'categ': ['trt', 'sex', 'ascites', 'hepato', 'spiders'],
            'criterion': 'peto', 'depth': 15, 'ens_metric_name': 'roc', 'max_features': 'sqrt',
            'min_samples_leaf': 1, 'mode_wei': 'square', 'n_estimators': 15, 'size_sample': 0.5},
    # Baseline time: 350,59 -> 77.4
    "ONK": {'aggreg_func': 'wei', 'categ': ['Диагноз'], 'criterion': 'peto',
            'depth': 15, 'ens_metric_name': 'conc', 'max_features': 'sqrt',
            'min_samples_leaf': 100, 'mode_wei': 'square', 'n_estimators': 30, 'size_sample': 0.5}
}

def get_samples(dataset="PBC"):
    X, y, features, categ, sch_nan = DATASETS_LOAD[dataset]()
    a = generate_sample(X, y, 5)
    return next(a)


def profile_tree(dataset="PBC", n_jobs=1):
    # params = {"criterion": "peto", "depth": 10, "min_samples_leaf": 1, "signif": 0.05, "n_jobs": n_jobs}
    params = TREE_PARAMS[dataset]
    params["n_jobs"] = n_jobs

    X_train, y_train, X_test, y_test, bins = get_samples(dataset=dataset)

    bins = get_bins(time=y_train[TIME_NAME], cens=y_train[CENS_NAME])
    profiler = cProfile.Profile()
    profiler.enable()
    for i in range(5):
        craid_tree = CRAID(**params)
        craid_tree.fit(X_train, y_train)
        craid_tree.predict_at_times(X_test, bins=bins, mode="surv")
        craid_tree.predict(X_test, target=TIME_NAME)
        craid_tree.predict_at_times(X_test, bins=bins, mode="hazard")

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    stats.dump_stats(f'profile_reports/{dataset}/tree_output_numba_{n_jobs}.pstats')


def profile_boost(dataset="PBC", n_jobs=1):
    # params = {"criterion": "peto", "depth": 5, "min_samples_leaf": 30, "n_estimators": 3, "n_jobs": n_jobs}
    params = BOOST_PARAMS[dataset]
    params["n_jobs"] = n_jobs

    X_train, y_train, X_test, y_test, bins = get_samples(dataset=dataset)
    profiler = cProfile.Profile()
    profiler.enable()
    for i in range(5):
        bst_model = BoostingCRAID(**params)
        bst_model.fit(X_train, y_train)
        bst_model.predict_at_times(X_test, bins=bins, mode="surv")
        bst_model.predict(X_test, target=TIME_NAME)
        bst_model.predict_at_times(X_test, bins=bins, mode="hazard")
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    stats.dump_stats(f'profile_reports/{dataset}/bst_output_numba_{n_jobs}.pstats')


def profile_exp():
    profiler = cProfile.Profile()
    profiler.enable()
    res_exp = run("PBC", with_self=["TREE"], with_external=False)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    stats.dump_stats('profile_reports/exp_output.pstats')


if __name__ == '__main__':
    # Visualize in browser html with command prompt "snakeviz file.pstats"
    # for n in [2**i for i in range(5)]:
    #     profile_tree(dataset="PBC", n_jobs=n)
    #     profile_tree(dataset="ONK", n_jobs=n)
    #     profile_boost(dataset="PBC", n_jobs=n)
    #     profile_boost(dataset="ONK", n_jobs=n)
    profile_exp()
