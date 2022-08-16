import cProfile
import pstats

from survivors.datasets import load_pbc_dataset
from survivors.experiments.grid import generate_sample

from survivors.tree import CRAID

def pbs_samples():
    X, y, features, categ, sch_nan = load_pbc_dataset()
    a = generate_sample(X, y, 5)
    return next(a)


if __name__ == '__main__':
    params = {"criterion": "peto", "depth": 10, "min_samples_leaf": 1, "signif": 0.05, "n_jobs": 1}
    X_train, y_train, X_test, y_test, bins = pbs_samples()

    profiler = cProfile.Profile()
    profiler.enable()
    craid_tree = CRAID(**params)
    craid_tree.fit(X_train, y_train)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    stats.dump_stats('profile_reports/output.pstats')
