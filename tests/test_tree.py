import numpy as np
import pytest
import os
import tempfile

from scipy.stats import chi2
from survivors.datasets import load_pbc_dataset
from survivors.experiments.grid import generate_sample

from survivors.criteria import chi2_sf, chi2_isf
from survivors.tree import CRAID
from survivors.ensemble import BoostingCRAID
from survivors.metrics import ibs


@pytest.mark.parametrize(
    ("scipy_func", "self_func", "space"),
    [(chi2.sf, chi2_sf, np.linspace(0.0, 10, 1000)),
     (chi2.isf, chi2_isf, np.linspace(0.0, 1, 1000))]
)
def chi2_deviation(scipy_func, self_func, space):
    for v in space:
        deviation = abs(scipy_func(v, df=1) - self_func(v, df=1))
        assert (deviation < 1e-10) or np.isnan(deviation)


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


# @pytest.mark.skip(reason="no way of currently testing this")
@pytest.mark.parametrize(
    ("params", "ibs_thres"),
    [({"criterion": "peto", "depth": 2, "min_samples_leaf": 30, "signif": 0.05}, 0.18),
     ({"criterion": "peto", "depth": 2, "min_samples_leaf": 30, "signif": 0.05, "cut": True}, 0.23)]
)
def test_tree(pbs_samples, params, ibs_thres):
    X_train, y_train, X_test, y_test, bins = pbs_samples
    craid_tree = CRAID(**params)
    craid_tree.fit(X_train, y_train)

    pred_time = craid_tree.predict(X_test, target="time")
    pred_prob = craid_tree.predict(X_test, target="cens")
    pred_sf = craid_tree.predict_at_times(X_test, bins=bins, mode="surv")
    assert ibs(y_train, y_test, pred_sf, bins, axis=-1) < ibs_thres


@pytest.mark.parametrize(
    ("params", "ibs_thres"),
    [({"criterion": "peto", "depth": 5, "min_samples_leaf": 30, "n_estimators": 3}, 0.19),
     ({"criterion": "weights", "depth": 5, "min_samples_leaf": 10, "n_estimators": 6}, 0.175)]
)
def test_boosting(pbs_samples, params, ibs_thres):
    X_train, y_train, X_test, y_test, bins = pbs_samples
    boost = BoostingCRAID(**params)
    boost.fit(X_train, y_train)
    pred_time = boost.predict(X_test, target="time")
    pred_prob = boost.predict(X_test, target="cens")
    pred_sf = boost.predict_at_times(X_test, bins=bins, mode="surv")
    assert ibs(y_train, y_test, pred_sf, bins, axis=-1) < ibs_thres


@pytest.mark.skip(reason="no way of currently testing this")
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
def _test_tree_visualize(pbs_samples, params, mode, size_expected):
    X_train, y_train, X_test, y_test, bins = pbs_samples
    craid_tree = CRAID(**params)
    craid_tree.fit(X_train, y_train)
    with tempfile.TemporaryDirectory() as tmp_dir:
        craid_tree.visualize(tmp_dir, mode=mode)
        stat_result = os.stat(os.path.join(tmp_dir, os.listdir(tmp_dir)[0]))
        assert stat_result.st_size == size_expected


@pytest.mark.skip(reason="no way of currently testing this")
@pytest.mark.parametrize(
    ("params", "depth_hist", "numb_hist"),
    [({"criterion": "peto", "depth": 5, "min_samples_leaf": 3, "signif": 0.05},
      np.array([4, 8, 0, 10, 62]), np.array([4, 8, 2, 8, 62])),
     ({"criterion": "peto", "depth": 5, "min_samples_leaf": 3, "signif": 0.05, "cut": True},
      np.array([24, 0, 0, 0, 60]), np.array([23, 1, 0, 0, 60])),
    ]
)
def _test_tree_predict_attr(pbs_samples, params, depth_hist, numb_hist):
    X_train, y_train, X_test, y_test, bins = pbs_samples
    craid_tree = CRAID(**params)
    craid_tree.fit(X_train, y_train)

    pred_depth = craid_tree.predict(X_test, target="depth")
    pred_numb = craid_tree.predict(X_test, target="numb")
    assert (np.histogram(pred_depth, bins=5)[0] == depth_hist).all()
    assert (np.histogram(pred_numb, bins=5)[0] == numb_hist).all()


@pytest.mark.skip(reason="no way of currently testing this")
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


@pytest.mark.skip(reason="no way of currently testing this")
def _test_tree_scheme(pbs_samples):
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
