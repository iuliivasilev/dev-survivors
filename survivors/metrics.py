from sksurv.metrics import check_y_survival, CensoringDistributionEstimator
import numpy as np
from numba import njit, jit
from lifelines import KaplanMeierFitter, NelsonAalenFitter
# from lifelines.utils import concordance_index

from .constants import TIME_NAME, CENS_NAME

METRIC_DICT = {
    "CI": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        concordance_index(y_tst[TIME_NAME], pred_time),
    "CI_CENS": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        concordance_index(y_tst[TIME_NAME], pred_time, y_tst[CENS_NAME]),

    "IBS": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        ibs(y_tr, y_tst, pred_sf, bins),
    "BAL_IBS": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        bal_ibs(y_tr, y_tst, pred_sf, bins),

    "IBS_WW": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        ibs_WW(y_tr, y_tst, pred_sf, bins),
    "BAL_IBS_WW": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        bal_ibs_WW(y_tr, y_tst, pred_sf, bins),

    "IBS_REMAIN": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        ibs_remain(y_tr, y_tst, pred_sf, bins),
    "BAL_IBS_REMAIN": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        bal_ibs_remain(y_tr, y_tst, pred_sf, bins),

    "IAUC": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        iauc(y_tr, y_tst, pred_hf, bins),
    "IAUC_WW": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        iauc_WW(y_tr, y_tst, pred_hf, bins),
    "IAUC_TI": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        iauc_TI(y_tr, y_tst, pred_hf, bins),
    "IAUC_WW_TI": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        iauc_WW_TI(y_tr, y_tst, pred_hf, bins),

    "AUPRC": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        auprc(y_tr, y_tst, pred_sf, bins),
    "AUPRC_by_obs": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        auprc(y_tr, y_tst, pred_sf, bins, axis=3),
    "EVENT_AUPRC": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        event_auprc(y_tr, y_tst, pred_sf, bins),
    "CENS_AUPRC": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        cens_auprc(y_tr, y_tst, pred_sf, bins),
    "BAL_AUPRC": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        bal_auprc(y_tr, y_tst, pred_sf, bins),

    "LOGLIKELIHOOD": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        loglikelihood(y_tst[TIME_NAME], y_tst[CENS_NAME], pred_sf, pred_hf, bins),
    "KL": lambda y_tr, y_tst, pred_time, pred_sf, pred_hf, bins:
        kl(y_tst[TIME_NAME], y_tst[CENS_NAME], pred_sf, pred_hf, bins)
}
""" dict: Available metrics in library and its realization """

DESCEND_METRICS = ['ibs', 'IBS', 'aic', "AIC", "bic", "BIC", "KL",
                   "BAL_IBS", "IBS_WW", "BAL_IBS_WW", "IBS_REMAIN", "BAL_IBS_REMAIN"]
""" list: Metrics with decreasing quality improvement """


def auprc(survival_train, survival_test, estimate, times, axis=-1):
    time = survival_test["time"]
    event = survival_test["cens"]

    steps = np.linspace(1e-5, 1 - 1e-5, 100)
    before_time = np.dot(time[:, np.newaxis], steps[np.newaxis, :])
    after_time = np.dot(time[:, np.newaxis], 1 / steps[np.newaxis, :])  # TODO OLD

    before_ind = np.clip(np.searchsorted(times, before_time), 0, times.shape[0] - 1)
    after_ind = np.clip(np.searchsorted(times, after_time), 0, times.shape[0] - 1)

    est = np.take_along_axis(estimate, before_ind, axis=1)
    est[event] -= np.take_along_axis(estimate[event], after_ind[event], axis=1)

    if axis == -1:  # mean for each time and observation
        est = np.mean(est, axis=0)  # TODO np.mean
        return np.trapz(est, steps)
        # return np.median(np.trapz(est, steps))
    elif axis == 0:  # for each observation
        return np.trapz(est, steps)
    elif axis == 1:  # in time (for graphics)
        # est = est.median(axis=0)
        est = est.mean(axis=0)  # TODO np.mean
        return est
    elif axis == 2:  # source
        return est
    elif axis == 3:  # for each observation with array wrap
        return np.array([np.trapz(est, steps)])
    return None


def bal_auprc(survival_train, survival_test, estimate, times, axis=-1):
    auprc_event = auprc(survival_train, survival_test[survival_test["cens"]],
                        estimate[survival_test["cens"]], times, axis=axis)
    auprc_cens = auprc(survival_train, survival_test[~survival_test["cens"]],
                       estimate[~survival_test["cens"]], times, axis=axis)
    return (auprc_event + auprc_cens) / 2


def event_auprc(survival_train, survival_test, estimate, times, axis=-1):
    auprc_event = auprc(survival_train, survival_test[survival_test["cens"]],
                        estimate[survival_test["cens"]], times, axis=axis)
    return auprc_event


def cens_auprc(survival_train, survival_test, estimate, times, axis=-1):
    auprc_cens = auprc(survival_train, survival_test[~survival_test["cens"]],
                       estimate[~survival_test["cens"]], times, axis=axis)
    return auprc_cens


@njit
def get_before(estim, wei):
    return np.square(estim) * wei


@njit
def get_after(estim, prob_cens):
    return np.square(1 - estim) / prob_cens


def ibs(survival_train, survival_test, estimate, times, axis=-1):
    """
    Modified integrated brier score from scikit-survival (add axis)
    Modification: with axis = 0 count ibs for each observation

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.
    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.
    estimate : array-like, shape = (n_samples, n_times)
        Estimated risk of experiencing an event for test data at `times`.
        The i-th column must contain the estimated probability of
        remaining event-free up to the i-th time point.
    times : array-like, shape = (n_times,)
        The time points for which to estimate the Brier score.
    axis : int, optional
        With axis = 1 count bs for each time from times
        With axis = 0 count ibs for each observation.
        With axis = -1 count mean ibs.
        The default is -1.

    Returns
    -------
    ibs_value : float or array-like or None
        if axis = 0 return array of ibs for each observation
           axis = 1 return array of bs for each time from times
           axis = -1 return float
        else
            None

    """
    test_event, test_time = check_y_survival(survival_test, allow_all_censored=True)
    # estimate, times = _check_estimate_2d(estimate, test_time, times)
    estimate = np.array(estimate)
    if estimate.ndim == 1 and times.shape[0] == 1:
        estimate = estimate.reshape(-1, 1)
    estimate[estimate == -np.inf] = 0
    estimate[estimate == np.inf] = 0
    # fit IPCW estimator
    cens = CensoringDistributionEstimator().fit(survival_train)
    # calculate inverse probability of censoring weight at current time point t.
    prob_cens_t = cens.predict_proba(times)
    prob_cens_t[prob_cens_t == 0] = np.inf
    # calculate inverse probability of censoring weights at observed time point
    prob_cens_y = cens.predict_proba(test_time)
    prob_cens_y[prob_cens_y == 0] = np.inf
    wei = test_event / prob_cens_y
    
    estim_before = get_before(estimate, wei[np.newaxis, :].T)
    estim_after = get_after(estimate, prob_cens_t)
    brier_scores = np.array([np.where(test_time <= t, 
                                      estim_before[:, i], 
                                      estim_after[:, i])
                             for i, t in enumerate(times)])
    time_diff = times[-1] - times[0] if times[-1] > times[0] else 1
    if axis == -1:  # mean ibs for each time and observation
        brier_scores = np.mean(brier_scores, axis=1)  # TODO np.mean
        return np.trapz(brier_scores, times) / time_diff
        # return np.median(np.trapz(brier_scores, times, axis=0)) / time_diff
    elif axis == 0:  # ibs for each observation
        return np.trapz(brier_scores, times, axis=0) / time_diff
    elif axis == 1:  # bs in time (for graphics)
        # return np.median(brier_scores, axis=1)
        return np.mean(brier_scores, axis=1)  # TODO np.mean
    return None


def bal_ibs(survival_train, survival_test, estimate, times, axis=-1):
    """ IBS with equal impact of each event type """
    ibs_event = ibs(survival_train, survival_test[survival_test["cens"]],
                    estimate[survival_test["cens"]], times, axis=axis)
    ibs_cens = ibs(survival_train, survival_test[~survival_test["cens"]],
                   estimate[~survival_test["cens"]], times, axis=axis)
    return ibs_event + ibs_cens


def ibs_WW(survival_train, survival_test, estimate, times, axis=-1):
    """ IBS with equal impact of partial observation """
    test_event, test_time = check_y_survival(survival_test, allow_all_censored=True)
    estimate = np.array(estimate)
    if estimate.ndim == 1 and times.shape[0] == 1:
        estimate = estimate.reshape(-1, 1)
    estimate[estimate == -np.inf] = 0
    estimate[estimate == np.inf] = 0

    estim_before = np.square(estimate) * test_event[np.newaxis, :].T
    estim_after = np.square(1 - estimate)
    brier_scores = np.array([np.where(test_time <= t,
                                      estim_before[:, i],
                                      estim_after[:, i])
                             for i, t in enumerate(times)])
    time_diff = times[-1] - times[0] if times[-1] > times[0] else 1
    if axis == -1:  # mean ibs for each time and observation
        brier_scores = np.mean(brier_scores, axis=1)  # TODO np.mean
        return np.trapz(brier_scores, times) / time_diff
        # return np.median(np.trapz(brier_scores, times, axis=0)) / time_diff
    elif axis == 0:  # ibs for each observation
        return np.trapz(brier_scores, times, axis=0) / time_diff
    elif axis == 1:  # bs in time (for graphics)
        # return np.median(brier_scores, axis=1)
        return np.mean(brier_scores, axis=1)  # TODO np.mean
    return None


def bal_ibs_WW(survival_train, survival_test, estimate, times, axis=-1):
    """ IBS with equal impact of each event type and partial observation """
    ibs_event = ibs_WW(survival_train, survival_test[survival_test["cens"]],
                       estimate[survival_test["cens"]], times, axis=axis)
    ibs_cens = ibs_WW(survival_train, survival_test[~survival_test["cens"]],
                      estimate[~survival_test["cens"]], times, axis=axis)
    return ibs_event + ibs_cens


def ibs_remain(survival_train, survival_test, estimate, times, axis=-1):
    """ IBS with equal impact of partial observation with controlled quantity """
    test_event, test_time = check_y_survival(survival_test, allow_all_censored=True)
    estimate = np.array(estimate)
    if estimate.ndim == 1 and times.shape[0] == 1:
        estimate = estimate.reshape(-1, 1)
    estimate[estimate == -np.inf] = 0
    estimate[estimate == np.inf] = 0

    estim_before = np.square(estimate) * test_event[np.newaxis, :].T
    estim_after = np.square(1 - estimate)
    brier_scores = np.array([np.where(test_time < t,
                                      estim_before[:, i],
                                      estim_after[:, i])
                             for i, t in enumerate(times)])
    N = np.sum(np.array([np.where(test_time < t, test_event, 1)
                         for i, t in enumerate(times)]), axis=1)
    # ind = np.digitize(test_time, times)
    # n_cens = np.bincount(ind[~test_event], minlength=times.shape[0])
    #
    # N = np.ones(times.shape) * np.sum(test_event)
    # if n_cens.shape[0] > 0:
    #     N += np.cumsum(n_cens[::-1])[::-1]
    time_diff = times[-1] - times[0] if times[-1] > times[0] else 1
    if axis == -1:  # mean ibs for each time and observation
        # brier_scores = np.mean(brier_scores, axis=1)
        brier_scores = np.where(N > 0, 1 / N, 0) * np.sum(brier_scores, axis=1)
        return np.trapz(brier_scores, times) / time_diff
    elif axis == 0:  # ibs for each observation
        return np.trapz(brier_scores, times, axis=0) / time_diff
    elif axis == 1:  # bs in time (for graphics)
        # brier_scores = np.mean(brier_scores, axis=1)
        brier_scores = np.where(N > 0, 1 / N, 0) * np.sum(brier_scores, axis=1)
        return brier_scores
    return None


def bal_ibs_remain(survival_train, survival_test, estimate, times, axis=-1):
    """ IBS with equal impact of each event type and partial observation with controlled quantity """
    ibs_event = ibs_remain(survival_train, survival_test[survival_test["cens"]],
                           estimate[survival_test["cens"]], times, axis=axis)
    ibs_cens = ibs_remain(survival_train, survival_test[~survival_test["cens"]],
                          estimate[~survival_test["cens"]], times, axis=axis)
    return ibs_event + ibs_cens


def iauc(survival_train, survival_test, estimate, times, tied_tol=1e-8, no_wei=False, time_int=False):
    """
    Modified integrated AUC (cumulative_dynamic_auc) 
        from scikit-survival (reduce complexity)

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.
    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.
    estimate : array-like, shape = (n_samples, n_times)
        Estimated risk of experiencing an event for test data at `times`.
        The i-th column must contain the estimated probability of
        remaining event-free up to the i-th time point.
    times : array-like, shape = (n_times,)
        The time points for which to estimate the Brier score.
    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.
        
    Returns
    -------
    mean_auc : float
        Summary measure referring to the mean cumulative/dynamic AUC
        over the specified time range `(times[0], times[-1])`.

    """
    if survival_train[CENS_NAME].sum() == 0:
        survival_train[CENS_NAME] = 1
        survival_test[CENS_NAME] = 1 - survival_test[CENS_NAME]
    if survival_test[CENS_NAME].sum() == 0:
        survival_test[CENS_NAME] = 1
    test_event, test_time = check_y_survival(survival_test)
    # estimate, times = _check_estimate_2d(estimate, test_time, times)
    estimate = np.array(estimate)
    n_samples = estimate.shape[0]
    n_times = times.shape[0]
    if estimate.ndim == 1:
        estimate = np.broadcast_to(estimate[:, np.newaxis], (n_samples, n_times))

    # fit and transform IPCW
    if no_wei:
        ipcw = np.ones(test_time.shape[0])
    else:
        cens = CensoringDistributionEstimator()
        cens.fit(survival_train)
        g_hat = cens.predict_proba(test_time[test_event])
        ipcw = np.zeros(test_time.shape[0])
        g_hat[g_hat == 0] = np.inf
        if not ((g_hat == 0.0).any()):
            ipcw[test_event] = 1.0 / g_hat
        else:
            ipcw = np.ones(test_time.shape[0])

    # expand arrays to (n_samples, n_times) shape
    test_time = np.broadcast_to(test_time[:, np.newaxis], (n_samples, n_times))
    test_event = np.broadcast_to(test_event[:, np.newaxis], (n_samples, n_times))
    times_2d = np.broadcast_to(times, (n_samples, n_times))
    ipcw = np.broadcast_to(ipcw[:, np.newaxis], (n_samples, n_times))

    # sort each time point (columns) by risk score (descending)
    o = np.argsort(-estimate, axis=0)
    test_time = np.take_along_axis(test_time, o, axis=0)
    test_event = np.take_along_axis(test_event, o, axis=0)
    estimate = np.take_along_axis(estimate, o, axis=0)
    ipcw = np.take_along_axis(ipcw, o, axis=0)

    is_case = (test_time <= times_2d) & test_event
    is_control = test_time > times_2d
    n_controls = is_control.sum(axis=0)
    n_controls[n_controls == 0] = 1

    # prepend row of infinity values
    estimate_diff = np.concatenate((np.broadcast_to(np.inf, (1, n_times)), estimate))
    is_tied = np.absolute(np.diff(estimate_diff, axis=0)) <= tied_tol

    cumsum_tp = np.cumsum(is_case * ipcw, axis=0)
    cumsum_fp = np.cumsum(is_control, axis=0)
    true_pos = cumsum_tp / cumsum_tp[-1]
    false_pos = cumsum_fp / n_controls

    scores = np.empty(n_times, dtype=float)
    it = np.nditer((true_pos, false_pos, is_tied), order="F", flags=["external_loop"])
    with it:
        for i, (tp, fp, mask) in enumerate(it):
            idx = np.flatnonzero(mask) - 1
            # only keep the last estimate for tied risk scores
            tp_no_ties = np.delete(tp, idx)
            fp_no_ties = np.delete(fp, idx)
            # Add an extra threshold position
            # to make sure that the curve starts at (0, 0)
            tp_no_ties = np.r_[0, tp_no_ties]
            fp_no_ties = np.r_[0, fp_no_ties]
            scores[i] = np.trapz(tp_no_ties, fp_no_ties)

    scores[np.isnan(scores)] = 0
    if n_times != 1:
        if time_int:
            time_diff = times[-1] - times[0] if times[-1] > times[0] else 1
            return np.trapz(scores, times, axis=0) / time_diff
        else:
            surv = KaplanMeierFitter()
            surv.fit(survival_test[TIME_NAME], survival_test[CENS_NAME])
            s_times = surv.survival_function_at_times(times).to_numpy()

            # compute integral of AUC over survival function
            d = -np.diff(np.r_[1.0, s_times])
            integral = (scores * d).sum()
            return integral / (1.0 - s_times[-1])
    return scores[0]


def iauc_WW(s_tr, s_tst, est, times, tied_tol=1e-8):
    """ IAUC without weighting for each observation """
    return iauc(s_tr, s_tst, est, times, tied_tol=tied_tol, no_wei=True)


def iauc_TI(s_tr, s_tst, est, times, tied_tol=1e-8):
    """ IAUC with integration by time (instead of S(t)) """
    return iauc(s_tr, s_tst, est, times, tied_tol=tied_tol, no_wei=False, time_int=True)


def iauc_WW_TI(s_tr, s_tst, est, times, tied_tol=1e-8):
    """ IAUC with equal weight for each observation and integration by time """
    return iauc(s_tr, s_tst, est, times, tied_tol=tied_tol, no_wei=True, time_int=True)


def ipa(survival_train, survival_test, estimate, times, axis=-1):
    """
    Index of Prediction Accuracy: General R^2 for binary outcome and right
    censored time to event (survival) outcome also with competing risks.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.
    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.
    estimate : array-like, shape = (n_samples, n_times)
        Estimated risk of experiencing an event for test data at `times`.
        The i-th column must contain the estimated probability of
        remaining event-free up to the i-th time point.
    times : array-like, shape = (n_times,)
        The time points for which to estimate the Brier score.
    axis : int, optional
        With axis = 1 count ipa for each time from times
        With axis = 0 count ipa for each observation.
        With axis = -1 count mean ipa.
        The default is -1.

    Returns
    -------
    ibs_value : float or array-like or None
        if axis = 0 return array
           axis = 1 return array
           axis = -1 return float
        else
            None

    """
    one_sf = get_survival_func(survival_train['time'], survival_train['cens'], times)[np.newaxis, :]
    kmf_estimate = np.repeat(one_sf, survival_test.shape[0], axis=0)

    ibs_model = ibs(survival_train, survival_test, estimate, times, axis)
    ibs_kmf_model = ibs(survival_train, survival_test, kmf_estimate, times, axis)
    return 1 - (ibs_model + 1e-8) / (ibs_kmf_model + 1e-8)


def kl(time, cens, sf, cumhf, bins):
    """ Kullback-Leibler divergence in terms of survival analysis """
    eq_sf = np.array([np.where(time > t, 1, 0)
                      for i, t in enumerate(bins)]).T

    kl_v = np.sum(sf * np.log((sf + 1e-10) / (eq_sf + 1e-10)) + np.abs(sf - eq_sf), axis=0)
    return np.trapz(kl_v, bins, axis=0) / (bins[-1] - bins[0])


def loglikelihood(time, cens, sf, cumhf, bins):
    """ Likelihood in terms of survival analysis (without PH) """
    index_times = np.digitize(time, bins, right=True) - 1
    hf = np.hstack((cumhf[:, 0][np.newaxis].T, np.diff(cumhf)))
    sf_by_times = np.take_along_axis(sf, index_times[:, np.newaxis], axis=1)[:, 0] + 1e-10
    hf_by_times = (np.take_along_axis(hf, index_times[:, np.newaxis], axis=1)[:, 0] + 1e-10)**cens
    likelihood = np.sum(np.log(sf_by_times) + np.log(hf_by_times))
    return likelihood


def aic(num_params, time, cens, sf, cumhf, bins):
    return 2*num_params - 2*loglikelihood(time, cens, sf, cumhf, bins)


def bic(k, n, time, cens, sf, cumhf, bins):
    return k*np.log(n) - 2*loglikelihood(time, cens, sf, cumhf, bins)


@njit
def count_pairs(T, P, E):
    n = len(T)
    concordant_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if E[i] == 1 and T[i] <= T[j]:
                total_pairs += 1
                concordant_pairs += P[i] < P[j]
                concordant_pairs += 0.5 * (P[i] == P[j])
    return concordant_pairs, total_pairs


def concordance_index(event_times, predicted_scores, event_observed=None):
    """
    Calculates the concordance index (C-index) for survival analysis.
    Previous speed: 684 ms ± 4.06 ms per loop (mean ± std. dev. of 7 runs, 1 loop each).
    Current speed: 22.7 ms ± 128 µs per loop (mean ± std. dev. of 7 runs, 10 loops each).
    Speed has increased by ~30 times

    Parameters
    ----------
    event_times: Array of true event times.
    predicted_scores: Array of predicted event times.
    event_observed: Array of event indicators (1 if event occurred, 0 if censored).

    Returns
    -------
    The concordance index: float

    Examples
    --------
    .. code:: python

        from survivors.metrics import concordance_index
        concordance_index(np.array([10, 20, 30, 40]),
                          np.array([20, 19, 29, 39]),
                          np.array([1, 0, 1, 0]))
    """
    if event_observed is None:
        event_observed = np.ones(len(event_times))
    order = np.argsort(event_times)
    predicted_scores = np.asarray(predicted_scores)[order]
    event_times = np.asarray(event_times)[order]
    event_observed = np.asarray(event_observed)[order]
    concordant_pairs, total_pairs = count_pairs(event_times, predicted_scores, event_observed)

    if total_pairs == 0:
        return 0
    return concordant_pairs / total_pairs


""" ESTIMATE FUNCTION """


def get_survival_func(ddeath, cdeath, bins=None):
    """
    Build Kaplan-Meier Estimate of survival function

    Parameters
    ----------
    ddeath : array-like
        Times of occurred events
    cdeath : array-like
        Indicate of occurred events (Censoring flag)
    bins : array-like, optional
        Points of survival function. The default is None.

    Returns
    -------
    KaplanMeierFitter or array
        If bins is None return kaplan-meier model, 
                   else return values of SF.

    """
    kmf = KaplanMeierFitter()
    kmf.fit(ddeath, cdeath)
    if not (bins is None):
        return kmf.survival_function_at_times(bins).to_numpy()
    return kmf


def get_hazard_func(ddeath, cdeath, bins=None):
    """
    Build Nelson-Aalen Estimate of Hazard function

    Parameters
    ----------
    ddeath : array-like
        Times of occurred events
    cdeath : array-like
        Indicate of occurred events (Censoring flag)
    bins : array-like, optional
        Points of hazard function. The default is None.

    Returns
    -------
    NelsonAalenFitter or array
        If bins is None return Nelson-Aalen model, 
                   else return values of HF.

    """
    naf = NelsonAalenFitter()
    naf.fit(ddeath, cdeath)
    if not (bins is None):
        return naf.cumulative_hazard_at_times(bins).to_numpy()
    return naf


IBS_DICT = {
    m.__name__.upper(): m
    for m in [ibs, bal_ibs,
              ibs_WW, bal_ibs_WW,
              ibs_remain, bal_ibs_remain]
}
