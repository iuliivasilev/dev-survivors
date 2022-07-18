import time


def info(function):
    def wrapped(*args):
        start_time = time.perf_counter()
        res = function(*args)
        exec_time = time.perf_counter() - start_time
        print(f"FUNCTION: {function.__name__} = OK, TIME: {exec_time}s")
        return res
    return wrapped


@info
def test_pbc_tree(X_train, y_train, X_test, y_test, bins):
    from survivors.tree import CRAID
    craid_tree = CRAID(criterion="peto", depth=2, min_samples_leaf=30, signif=0.05)
    craid_tree.fit(X_train, y_train)

    pred_time = craid_tree.predict(X_test, target="time")
    pred_prob = craid_tree.predict(X_test, target="cens")
    pred_surv = craid_tree.predict_at_times(X_test, bins=bins, mode="surv")

    i = 12
    assert y_test[i]['cens'] == False
    assert y_test[i]['time'] == 3577.0
    assert round(pred_prob[i], 5) == 0.17045
    assert round(pred_time[i], 5) == 1971.20455

    assert round(pred_surv[i][0], 5) == 1.0
    assert round(pred_surv[i][-1], 5) == 0.32086
    assert round(pred_surv[i].mean(), 5) == 0.79905


@info
def test_pbc_boosting(X_train, y_train, X_test, y_test, bins):
    from survivors.ensemble import BoostingCRAID

    bstr = BoostingCRAID(criterion="peto", depth=5, min_samples_leaf=30, n_estimators=3)
    bstr.fit(X_train, y_train)
    pred_time = bstr.predict(X_test, target="time")
    pred_prob = bstr.predict(X_test, target="cens")
    pred_surv = bstr.predict_at_times(X_test, bins=bins, mode="surv")

    assert bstr.bettas == [0.06726457733777294, 0.08173060857576495, 0.048712693796604756]
    assert round(bstr.weights[0], 5) == 0.10381
    assert round(bstr.weights[-1], 5) == 0.06726

    i = 12
    assert round(pred_prob[i], 5) == 0.13312
    assert round(pred_time[i], 5) == 1908.03485
    assert round(pred_surv[i][0], 5) == 1.0
    assert round(pred_surv[i][-1], 5) == 0.74072
    assert round(pred_surv[i].mean(), 5) == 0.87483


@info
def test_pbc_dataset():
    from survivors.datasets import load_pbc_dataset
    from survivors.experiments.grid import generate_sample

    X, y, features, categ, sch_nan = load_pbc_dataset()

    # Train contains 80% of source data
    # Test contains 20% of source data
    a = generate_sample(X, y, 5)
    X_train, y_train, X_test, y_test, bins = next(a)
    test_pbc_tree(X_train, y_train, X_test, y_test, bins)
    test_pbc_boosting(X_train, y_train, X_test, y_test, bins)


if __name__ == "__main__":
    test_pbc_dataset()
