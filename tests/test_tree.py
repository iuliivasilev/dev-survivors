import numpy as np
import pytest

from survivors.datasets import load_pbc_dataset
from survivors.experiments.grid import generate_sample

from survivors.tree import CRAID
from survivors.ensemble import BoostingCRAID


@pytest.fixture(scope="module")
def pbs_samples():
    X, y, features, categ, sch_nan = load_pbc_dataset()
    a = generate_sample(X, y, 5)
    return next(a)

# import time
# def info(function):
#     def wrapped(**kwargs):
#         start_time = time.perf_counter()
#         res = function(**kwargs)
#         exec_time = time.perf_counter() - start_time
#         print(f"FUNCTION: {function.__name__} = OK, TIME: {exec_time}s")
#         return res
#     return wrapped


@pytest.mark.parametrize(
    ("params", "n_obser", "l_expected"),
    [({"criterion": "peto", "depth": 2, "min_samples_leaf": 30, "signif": 0.05},
     12, [0.17045, 1971.20455, 1.0, 0.32086, 0.79905]),
     ({"criterion": "peto", "depth": 2, "min_samples_leaf": 30, "signif": 0.05, "cut": True},
      12, [0.10588, 2069.91765, 1.0, 0.83601, 0.90273])
    ]
)
def test_tree(pbs_samples, params, n_obser, l_expected):
    X_train, y_train, X_test, y_test, bins = pbs_samples
    craid_tree = CRAID(**params)
    craid_tree.fit(X_train, y_train)

    pred_time = craid_tree.predict(X_test, target="time")
    pred_prob = craid_tree.predict(X_test, target="cens")
    pred_surv = craid_tree.predict_at_times(X_test, bins=bins, mode="surv")

    assert round(pred_prob[n_obser], 5) == l_expected[0]
    assert round(pred_time[n_obser], 5) == l_expected[1]
    assert round(pred_surv[n_obser][0], 5) == l_expected[2]
    assert round(pred_surv[n_obser][-1], 5) == l_expected[3]
    assert round(pred_surv[n_obser].mean(), 5) == l_expected[4]


@pytest.mark.parametrize(
    ("params", "n_obser", "l_expected", "boost_bettas"),
    [({"criterion": "peto", "depth": 5, "min_samples_leaf": 30, "n_estimators": 3},
     12, [0.13312, 1908.03485, 1.0, 0.74072, 0.87483], [0.06726, 0.08173, 0.04871])
    ]
)
def test_boosting(pbs_samples, params, n_obser, l_expected, boost_bettas):
    X_train, y_train, X_test, y_test, bins = pbs_samples
    bstr = BoostingCRAID(**params)
    bstr.fit(X_train, y_train)
    pred_time = bstr.predict(X_test, target="time")
    pred_prob = bstr.predict(X_test, target="cens")
    pred_surv = bstr.predict_at_times(X_test, bins=bins, mode="surv")

    assert list(np.round(bstr.bettas, 5)) == boost_bettas
    assert round(pred_prob[n_obser], 5) == l_expected[0]
    assert round(pred_time[n_obser], 5) == l_expected[1]
    assert round(pred_surv[n_obser][0], 5) == l_expected[2]
    assert round(pred_surv[n_obser][-1], 5) == l_expected[3]
    assert round(pred_surv[n_obser].mean(), 5) == l_expected[4]
