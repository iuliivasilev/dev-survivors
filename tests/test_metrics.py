import numpy as np
from survivors.metrics import ibs, iauc, ipa, get_survival_func
from survivors.constants import get_y

from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
import pytest


def test_ibs():
    bins = np.array([1, 10, 100, 1000])

    y_train = get_y(np.array([1, 0]), np.array([100, 100]))
    y_test_1 = get_y(np.array([1]), np.array([100]))
    y_test_2 = get_y(np.array([0]), np.array([100]))
    y_test_3 = get_y(np.array([1]), np.array([50]))

    kmf_train = KaplanMeierFitter()
    kmf_train.fit(y_train['time'], event_observed=y_train['cens'])
    sf_train = kmf_train.survival_function_at_times(bins).to_numpy()[np.newaxis, :]

    assert ibs(y_train, y_test_1, sf_train, bins, axis=-1) == 0.0
    assert list(ibs(y_train, y_test_1, sf_train, bins, axis=0)) == [0.0]
    assert list(ibs(y_train, y_test_1, sf_train, bins, axis=1)) == [0.0, 0.0, 0.0, 0.0]
    assert ibs(y_train, y_test_2, sf_train, bins) == 0.0
    assert round(ibs(y_train, y_test_3, sf_train, bins), 5) == 0.23649
    assert list(np.round(ibs(y_train, y_test_3, sf_train, bins, axis=0), 5)) == [0.23649]
    assert list(np.round(ibs(y_train, y_test_3, sf_train, bins, axis=1), 5)) == [0.0, 0.0, 0.25, 0.25]


def test_iauc():
    bins = np.array([1, 10, 100, 1000])
    y_tr_gen = get_y(np.array([1, 1, 1]), np.array([10, 100, 1000]))
    y_test_gen_1 = get_y(np.array([1, 1]), np.array([10, 100]))
    y_test_gen_2 = get_y(np.array([1, 1, 1]), np.array([10, 100, 300]))

    naf_train = NelsonAalenFitter()
    naf_train.fit(y_tr_gen['time'], event_observed=y_tr_gen['cens'])
    haz_train = naf_train.cumulative_hazard_at_times(bins).to_numpy()

    y_test_est_1 = np.array([haz_train, haz_train * 2])
    y_test_est_2_1 = np.array([haz_train, haz_train, haz_train])
    y_test_est_2_2 = np.array([haz_train * 3, haz_train * 2, haz_train])

    assert round(iauc(y_tr_gen, y_test_gen_1, y_test_est_1, bins), 5) == 0.0
    assert round(iauc(y_tr_gen, y_test_gen_2, y_test_est_2_1, bins), 5) == 0.33333
    assert round(iauc(y_tr_gen, y_test_gen_2, y_test_est_2_2, bins), 5) == 0.66667


def test_ipa():
    def kmf_estimate_generation(y_train, y_test):
        sf_train = get_survival_func(np.hstack([y_train['time'], y_test['time']]),
                                     np.hstack([y_train['cens'], y_test['cens']]), bins)[np.newaxis, :]
        return np.repeat(sf_train, y_test.shape[0], axis=0)

    bins = np.array([1, 10, 100, 1000])
    y_tr_gen = get_y(np.array([1, 0]), np.array([100, 200]))
    y_test_1 = get_y(np.array([1]), np.array([100]))
    y_test_2 = get_y(np.array([0]), np.array([100]))
    y_test_3 = get_y(np.array([1, 0, 1]), np.array([50, 100, 200]))

    estim_1 = kmf_estimate_generation(y_tr_gen, y_test_1)
    estim_2 = kmf_estimate_generation(y_tr_gen, y_test_2)
    estim_3 = kmf_estimate_generation(y_tr_gen, y_test_3)

    assert round(ipa(y_tr_gen, y_test_1, estim_1, bins, axis=-1), 5) == 0.55556
    assert round(ipa(y_tr_gen, y_test_2, estim_2, bins, axis=-1), 5) == 0.0

    assert round(ipa(y_tr_gen, y_test_3, estim_3, bins, axis=-1), 5) == 0.1725
    assert list(np.round(ipa(y_tr_gen, y_test_3, estim_3, bins, axis=0), 5)) == [0.07429, 0.0, 0.36]
    assert list(np.round(ipa(y_tr_gen, y_test_3, estim_3, bins, axis=1), 5)) == [0.0, 0.0, -0.04, 0.64]
