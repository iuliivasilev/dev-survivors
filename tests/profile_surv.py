import cProfile
import pstats

from survivors.datasets import load_pbc_dataset, load_onk_dataset
from survivors.experiments.grid import generate_sample

from test_experiments import run


from survivors.constants import get_bins, TIME_NAME, CENS_NAME
from survivors.tree import CRAID
from survivors.ensemble import BoostingCRAID


def get_samples():
    X, y, features, categ, sch_nan = load_pbc_dataset()
    a = generate_sample(X, y, 5)
    return next(a)


def profile_tree():
    # params = {"criterion": "peto", "depth": 2, "min_samples_leaf": 30, "signif": 0.05, "n_jobs": 1}
    params = {"criterion": "peto", "depth": 10, "min_samples_leaf": 1, "signif": 0.05, "n_jobs": 16}

    X_train, y_train, X_test, y_test, bins = get_samples()

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
    stats.dump_stats('profile_reports/PBC/tree_output_16.pstats')


def profile_boost():
    params = {"criterion": "peto", "depth": 5, "min_samples_leaf": 30, "n_estimators": 3, "n_jobs": 1}

    X_train, y_train, X_test, y_test, bins = get_samples()
    profiler = cProfile.Profile()
    profiler.enable()
    craid_tree = BoostingCRAID(**params)
    craid_tree.fit(X_train, y_train)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    stats.dump_stats('profile_reports/boost_output.pstats')


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
    profile_tree()
    # profile_exp()
