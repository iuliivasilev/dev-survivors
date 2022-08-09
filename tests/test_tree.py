import numpy as np
import pytest
import os
import tempfile

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


@pytest.mark.parametrize(
    ("params", "mode", "size_expected"),
    [({"criterion": "peto", "depth": 2, "min_samples_leaf": 30, "signif": 0.05}, "hist", 191784),
     ({"criterion": "peto", "depth": 2, "min_samples_leaf": 30, "signif": 0.05, "cut": True}, "surv", 282388),
     ({"criterion": "peto", "depth": 5, "min_samples_leaf": 30, "signif": 0.05}, "hist", 361978),
     ({"criterion": "peto", "depth": 5, "min_samples_leaf": 30, "signif": 0.05, "cut": True}, "surv", 381854),
     ({"criterion": "peto", "depth": 5, "min_samples_leaf": 1, "signif": 0.05}, "hist", 628687),
     ({"criterion": "peto", "depth": 5, "min_samples_leaf": 1, "signif": 0.05, "cut": True}, "surv", 685853)
    ]
)
def test_tree_visualize(pbs_samples, params, mode, size_expected):
    X_train, y_train, X_test, y_test, bins = pbs_samples
    craid_tree = CRAID(**params)
    craid_tree.fit(X_train, y_train)
    with tempfile.TemporaryDirectory() as tmp_dir:
        craid_tree.visualize(tmp_dir, mode=mode)
        stat_result = os.stat(os.path.join(tmp_dir, os.listdir(tmp_dir)[0]))
        assert stat_result.st_size == size_expected


def test_tree_rules(pbs_samples):
    params = {"criterion": "peto", "depth": 2, "min_samples_leaf": 30, "signif": 0.05}
    X_train, y_train, X_test, y_test, bins = pbs_samples
    craid_tree = CRAID(**params)
    craid_tree.fit(X_train, y_train)

    x = craid_tree.predict(X_test, mode="rules")
    a = np.array(np.unique(x, return_counts=True)).T
    assert (a[0] == np.array(['((bili >= 2.25)| nan) & ((protime >= 11.55)| nan)', 12], dtype=object)).all()
    assert (a[1] == np.array(['((bili >= 2.25)| nan) & (protime < 11.55)', 11], dtype=object)).all()
    assert (a[2] == np.array(['(bili < 2.25) & ((age >= 62.905)| nan)', 6], dtype=object)).all()
    assert (a[3] == np.array(['(bili < 2.25) & (age < 62.905)', 55], dtype=object)).all()


def test_tree_scheme(pbs_samples):
    params = {"criterion": "peto", "depth": 2, "min_samples_leaf": 30, "signif": 0.05}
    X_train, y_train, X_test, y_test, bins = pbs_samples
    craid_tree = CRAID(**params)
    craid_tree.fit(X_train, y_train)

    zero_feat_schemes = list(set(craid_tree.predict_schemes(X_test, [])))
    assert len(zero_feat_schemes) == 4
    zero_scheme_keys = [list(sch.schemes_dict.keys()) for sch in zero_feat_schemes]
    assert ['((bili >= 2.25)| nan) & (protime < 11.55)'] in zero_scheme_keys
    assert ['(bili < 2.25) & (age < 62.905)'] in zero_scheme_keys
    assert ['((bili >= 2.25)| nan) & ((protime >= 11.55)| nan)'] in zero_scheme_keys
    assert ['(bili < 2.25) & ((age >= 62.905)| nan)'] in zero_scheme_keys

    one_feat_schemes = list(set(craid_tree.predict_schemes(X_test, ["bili"])))
    assert len(one_feat_schemes) == 4
    one_scheme_keys = [list(sch.schemes_dict.keys()) for sch in one_feat_schemes]
    assert ['((bili >= 2.25)| nan) & (protime < 11.55)',
            '(bili < 2.25) & (age < 62.905)'] in one_scheme_keys
    assert ['((bili >= 2.25)| nan) & ((protime >= 11.55)| nan)',
            '(bili < 2.25) & ((age >= 62.905)| nan)'] in one_scheme_keys
    assert ['((bili >= 2.25)| nan) & ((protime >= 11.55)| nan)',
            '(bili < 2.25) & (age < 62.905)'] in one_scheme_keys
    assert ['((bili >= 2.25)| nan) & (protime < 11.55)',
            '(bili < 2.25) & ((age >= 62.905)| nan)'] in one_scheme_keys

    two_feat_schemes = list(set(craid_tree.predict_schemes(X_test, ["bili", "protime"])))
    assert len(two_feat_schemes) == 2
    two_scheme_keys = [list(sch.schemes_dict.keys()) for sch in two_feat_schemes]
    assert ['((bili >= 2.25)| nan) & ((protime >= 11.55)| nan)',
            '((bili >= 2.25)| nan) & (protime < 11.55)',
            '(bili < 2.25) & ((age >= 62.905)| nan)'] in two_scheme_keys
    assert ['((bili >= 2.25)| nan) & ((protime >= 11.55)| nan)',
            '((bili >= 2.25)| nan) & (protime < 11.55)',
            '(bili < 2.25) & (age < 62.905)'] in two_scheme_keys

    three_feat_schemes = list(set(craid_tree.predict_schemes(X_test, ["bili", "protime", "age"])))
    assert len(three_feat_schemes) == 1
    three_scheme_keys = [list(sch.schemes_dict.keys()) for sch in three_feat_schemes]
    assert ['((bili >= 2.25)| nan) & ((protime >= 11.55)| nan)',
            '((bili >= 2.25)| nan) & (protime < 11.55)',
            '(bili < 2.25) & ((age >= 62.905)| nan)',
            '(bili < 2.25) & (age < 62.905)'] in three_scheme_keys
