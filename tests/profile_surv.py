import cProfile
import pstats

from survivors.datasets import load_pbc_dataset
from survivors.experiments.grid import generate_sample

from test_experiments import run

from survivors.tree import CRAID
from survivors.ensemble import BoostingCRAID


def pbs_samples():
    X, y, features, categ, sch_nan = load_pbc_dataset()
    a = generate_sample(X, y, 5)
    return next(a)


def profile_tree():
    # params = {"criterion": "peto", "depth": 2, "min_samples_leaf": 30, "signif": 0.05, "n_jobs": 1}
    params = {"criterion": "peto", "depth": 10, "min_samples_leaf": 1, "signif": 0.05, "n_jobs": 8}

    X_train, y_train, X_test, y_test, bins = pbs_samples()
    profiler = cProfile.Profile()
    profiler.enable()
    craid_tree = CRAID(**params)
    craid_tree.fit(X_train, y_train)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    stats.dump_stats('profile_reports/tree_output.pstats')


def profile_boost():
    params = {"criterion": "peto", "depth": 5, "min_samples_leaf": 30, "n_estimators": 3, "n_jobs": 1}

    X_train, y_train, X_test, y_test, bins = pbs_samples()
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
    profile_exp()
