import numpy as np

from survivors.scheme import Scheme
from survivors.scheme import FilledSchemeStrategy
import pytest

@pytest.mark.skip(reason="no way of currently testing this")
def _test_scheme():
    a = Scheme(rule="(a > 0.5) & (b <= 0.5)",
               times=np.array([1, 2, 3, 4, 5]),
               cens=np.array([1, 0, 1, 0, 1]),
               feat_means=np.array([1, 10, 100, 1000]))
    b = Scheme(rule="(a > 0.5) & (b > 0.5)",
               times=np.array([6, 7, 8, 9, 10]),
               cens=np.array([1, 1, 1, 1, 1]),
               feat_means=np.array([10, 10, 10, 10]))
    c = Scheme(rule="(a <= 0.5) & (c <= 1000)",
               times=np.array([11, 12, 13, 14, 15]),
               cens=np.array([1, 1, 1, 1, 1]),
               feat_means=np.array([1, 1, 1, 1]))
    d = Scheme(rule="(a <= 0.5) & (c > 1000)",
               times=np.array([16, 17, 18, 19, 20]),
               cens=np.array([0, 1, 0, 0, 1]),
               feat_means=np.array([100, 100, 100, 100]))

    fss1 = FilledSchemeStrategy([a, b])
    fss2 = FilledSchemeStrategy([c, d])
    fss1.join(fss2)
    fss1.join_nearest_leaves(sign_thres=0.005)
    assert list(fss1.schemes_dict.keys()) == [
        '(a <= 0.5) & (c <= 1000)',
        '(a <= 0.5) & (c > 1000)',
        '((a > 0.5) & (b <= 0.5))|((a > 0.5) & (b > 0.5))'
    ]
    assert fss1.predict_best_scheme(sort_by="proba").rule == "(a <= 0.5) & (c <= 1000)"
    assert fss1.predict_best_scheme(sort_by="time").rule == "(a <= 0.5) & (c > 1000)"
    assert fss1.predict_best_scheme(sort_by="size").rule == "((a > 0.5) & (b <= 0.5))|((a > 0.5) & (b > 0.5))"

    # fss1.visualize_schemes(sort_by="size")
    # fss1.visualize_schemes(sort_by="size", os.getcwd() +'\\')
