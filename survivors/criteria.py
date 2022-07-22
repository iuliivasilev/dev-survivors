import numpy as np
import pandas as pd

from scipy import stats
from lifelines import CoxPHFitter
# from lifelines.statistics import logrank_test
# import fastlogranktest as flr
from numba import njit, jit

### Interface functions
def logrank(durations_A, durations_B, event_observed_A=None, event_observed_B=None):
    return weight_lr_fast(durations_A, durations_B, event_observed_A, event_observed_B)

def wilcoxon(durations_A, durations_B, event_observed_A=None, event_observed_B=None):
    return weight_lr_fast(durations_A, durations_B, event_observed_A, event_observed_B, weightings="wilcoxon")

def peto(durations_A, durations_B, event_observed_A=None, event_observed_B=None):
    return weight_lr_fast(durations_A, durations_B, event_observed_A, event_observed_B, weightings="peto")

def tarone_ware(durations_A, durations_B, event_observed_A=None, event_observed_B=None):
    return weight_lr_fast(durations_A, durations_B, event_observed_A, event_observed_B, weightings="tarone-ware")

CRITERIA_DICT = {
    "logrank": logrank,
    "wilcoxon": wilcoxon,
    "peto": peto,
    "tarone-ware": tarone_ware
}
""" dict: Available criteria in library and its realization """

def logrank_self(durations_A, durations_B, event_observed_A=None, event_observed_B=None) -> float:
    # 27.1 ms ± 1.26 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    try:
        event_times_A, event_times_B = (np.array(durations_A), np.array(durations_B))
        time = np.r_[event_times_A, event_times_B]
        group_indicator = np.r_[np.zeros(event_times_A.shape[0], dtype=int), np.ones(event_times_B.shape[0], dtype=int)]
        event = np.r_[event_observed_A, event_observed_B]
        n_samples = time.shape[0]
        groups, group_counts = np.unique(group_indicator, return_counts=True)
        n_groups = groups.shape[0]
        if n_groups == 1:
            raise ValueError("At least two groups must be specified, "
                             "but only one was provided.")

        # sort descending
        o = np.argsort(-time, kind="mergesort")
        x = group_indicator[o]
        event = event[o]
        time = time[o]

        at_risk = np.zeros(n_groups, dtype=np.int_)
        observed = np.zeros(n_groups, dtype=np.int_)
        expected = np.zeros(n_groups, dtype=np.float_)
        covar = np.zeros((n_groups, n_groups), dtype=np.float_)
        k = 0
        while k < n_samples:
            ti = time[k]
            total_events = 0
            while k < n_samples and ti == time[k]:
                idx = np.searchsorted(groups, x[k])
                if event[k]:
                    observed[idx] += 1
                    total_events += 1
                at_risk[idx] += 1
                k += 1

            if total_events != 0:
                total_at_risk = k
                expected += at_risk * (total_events / total_at_risk)
                if total_at_risk > 1:
                    multiplier = total_events * (total_at_risk - total_events) / (total_at_risk * (total_at_risk - 1))
                    for g1 in range(n_groups):
                        temp = at_risk[g1] * multiplier
                        covar[g1, g1] += temp
                        for g2 in range(n_groups):
                            covar[g1, g2] -= temp * at_risk[g2] / total_at_risk

        df = n_groups - 1
        zz = observed[:df] - expected[:df]
        chisq = np.linalg.solve(covar[:df, :df], zz).dot(zz)
        pval = stats.chi2.sf(chisq, df)
    except:
        pval = 1.0
    return pval
    
def cox(durations_A, durations_B, event_observed_A=None, event_observed_B=None) -> float:
    dfA = pd.DataFrame({'E': event_observed_A, 'T': durations_A, 'group': 0})
    dfB = pd.DataFrame({'E': event_observed_B, 'T': durations_B, 'group': 1})
    df_ = pd.concat([dfA, dfB])

    cph = CoxPHFitter().fit(df_, 'T', 'E')
    return cph.log_likelihood_ratio_test().p_value

### 06/02/2022 Numpy+Numba 
### accelerate 2.8 according to library criteria
@njit
def coeffs_t_j(dur_A, dur_B, cens_A, cens_B, t_j, weightings):
    N_1_j = (dur_A >= t_j).sum()#np.where(dur_A >= t_j,1,0).sum()
    N_2_j = (dur_B >= t_j).sum()#np.where(dur_B >= t_j,1,0).sum()
    if N_1_j == 0 or N_2_j == 0:
        return 0, 0, 0
    O_1_j = ((dur_A == t_j) * cens_A).sum() #np.where(dur_A == t_j, cens_A,0).sum()
    O_2_j = ((dur_B == t_j) * cens_B).sum()#np.where(dur_B == t_j, cens_B,0).sum()
    
    N_j = N_1_j + N_2_j
    O_j = O_1_j + O_2_j
    E_1_j = N_1_j*O_j/N_j
    w_j = 1
    if weightings == "wilcoxon":
        w_j = N_j
    elif weightings == "tarone-ware":
        w_j = np.sqrt(N_j)
    elif weightings == "peto":
        w_j = (1.0 - float(O_j)/(N_j+1))
    
    num = O_1_j - E_1_j
    denom = E_1_j*(N_j - O_j) * N_2_j/(N_j*(N_j - 1))
    return w_j, num, denom

@njit
def lr_statistic(dur_A, dur_B, cens_A, cens_B, times, weightings):
    res = np.zeros((times.shape[0], 3), dtype=np.float32)
    for j, t_j in enumerate(times):
        res[j] = coeffs_t_j(dur_A, dur_B, cens_A, cens_B, t_j, weightings)
    
    if weightings == "peto":
        res[:, 0] = np.cumprod(res[:, 0])
    logrank = np.power((res[:, 0]*res[:, 1]).sum(), 2) / ((res[:, 0]*res[:, 0]*res[:, 2]).sum())
    return logrank

def weight_lr_fast(dur_A, dur_B, cens_A = None, cens_B = None, weightings = ""):
    """
    Count weighted log-rank criteria

    Parameters
    ----------
    dur_A : array-like 
        Time of occured events from first sample.
    dur_B : array-like 
        Time of occured events from second sample.
    cens_A : array-like, optional
        Indicate of occured events from first sample. 
        The default is None (all events occured).
    cens_B : array-like, optional
        Indicate of occured events from second sample. 
        The default is None (all events occured).
    weightings : str, optional
        Weights of criteria. The default is "" (log-rank).
        Log-rank :math:'w = 1'
        Wilcoxon :math:'w = N_j'
        Tarone-ware :math:'w = \\sqrt(N_j)'
        Peto-peto :math:'w = \\fraq{1 - O_j}{N_j + 1}'

    Returns
    -------
    p-value : float
        Chi2 p-value of weighted log-rank test

    """
    try:
        if cens_A is None:
            cens_A = np.ones(dur_A.shape[0])
        if cens_B is None:
            cens_B = np.ones(dur_B.shape[0])

        #     a1 = np.unique(dur_A)
        #     a2 = np.unique(dur_B)
        #     times = np.unique(np.clip(np.union1d(a1,a2), 0, np.min([a1.max(), a2.max()])))
        times = np.union1d(np.unique(dur_A), np.unique(dur_B))
        logrank = lr_statistic(dur_A, dur_B, cens_A, cens_B, times, weightings)
        pvalue = stats.chi2.sf(logrank, df=1)
        return pvalue
    except:
        return 1.0