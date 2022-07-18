import numpy as np
from survivors.metrics import ibs, concordance_index, iauc
from survivors.constants import get_y

from lifelines import KaplanMeierFitter


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