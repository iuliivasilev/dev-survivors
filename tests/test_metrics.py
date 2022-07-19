import numpy as np
from survivors.metrics import ibs, concordance_index, iauc
from survivors.constants import get_y

from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter


def test_ibs():
    bins = np.array([1, 10, 100, 1000])

    y_train = get_y(np.array([1, 0]), np.array([100, 100]))
    y_test_1 = get_y(np.array([1]), np.array([100]))
    y_test_2 = get_y(np.array([0]), np.array([100]))
    y_test_3 = get_y(np.array([1]), np.array([50]))

    kmf_train = KaplanMeierFitter()
    kmf_train.fit(y_train['time'], event_observed=y_train['cens'])
    sf_train = kmf_train.survival_function_at_times(bins).to_numpy()[np.newaxis, :]

    assert ibs(y_train, y_test_1, sf_train, bins) == 0.0
    assert ibs(y_train, y_test_2, sf_train, bins) == 0.0
    assert round(ibs(y_train, y_test_3, sf_train, bins), 5) == 0.23649


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