import numpy as np
import math
from numba import njit, jit

# from scipy.stats import chi2
# from lifelines import CoxPHFitter
# from lifelines.statistics import logrank_test
# import fastlogranktest as flr

"""
SIGNIFICANCE EVALUATION (chi2.sf, chi2.isf)
MODIFIED CODE FROM 
1. https://github.com/etal/biofrills/blob/36684bb6c7632f96215e8b2b4ebc86640f331bcd/biofrills/stats/chisq.py
2. https://github.com/scipy/scipy/blob/v1.13.0/scipy/stats/_continuous_distns.py#L1547
3. https://github.com/scipy/scipy/blob/249c11f12ca11b000490da615c3008b2dd213322/scipy/special/special/cephes/igam.h
4. https://github.com/catboost/catboost/blob/f2c5eb6a1b1e1e936b56b3e523bd8dfd77b46fb9/contrib/python/scipy/py3/scipy/special/cephes/lanczos.h#L69
"""

MACHEP = 1e-13  # the machine roundoff error / tolerance
BORDER = 4e15
BORDERINV = 1 / BORDER

EULER = 0.577215664
MAXLOG = 709.79
LANCZOS_G = 6.02468

MAXITER = 2000
LGAMMA_05 = 0.5723649429247004


@njit
def _igamc(a, x):
    """
    In this implementation both arguments must be positive.
    The integral is evaluated by either a power series or
    continued fraction expansion, depending on the relative
    values of a and x.
    """
    # Compute  x**a * exp(-x) / Gamma(a)
    ax = math.exp(a * math.log(x) - x - LGAMMA_05)

    # Continued fraction
    y = 1.0 - a
    z = x + y + 1.0
    c = 0.0
    pkm2 = 1.0
    qkm2 = x
    pkm1 = x + 1.0
    qkm1 = z * x
    ans = pkm1 / qkm1
    while True:
        c += 1.0
        y += 1.0
        z += 2.0
        yc = y * c
        pk = pkm1 * z - pkm2 * yc
        qk = qkm1 * z - qkm2 * yc
        if qk != 0:
            r = pk / qk
            t = abs((ans - r) / r)
            ans = r
        else:
            t = 1.0
        pkm2 = pkm1
        pkm1 = pk
        qkm2 = qkm1
        qkm1 = qk
        if abs(pk) > BORDER:
            pkm2 *= BORDERINV
            pkm1 *= BORDERINV
            qkm2 *= BORDERINV
            qkm1 *= BORDERINV
        if t <= MACHEP:
            return ans * ax


@njit
def power_series(a, x):
    r = a
    c = 1.0
    ans = 1.0

    for i in range(0, MAXITER):
        r += 1.0
        c *= x / r
        ans += c
        if c <= MACHEP * ans:
            break
    return ans / a


@njit
def _igam(a, x):
    """ Left tail of incomplete Gamma function """
    # Compute  x**a * exp(-x) / Gamma(a)
    ax = math.exp(a * math.log(x) - x - LGAMMA_05)
    return power_series(a, x) * ax


@njit
def igam_series(a, x):
    ax = igam_fac(a, x)
    if ax == 0.0:
        return 0.0
    return power_series(a, x) * ax


@njit
def chi2_sf(x, df):
    """
    Probability value (1-tail) for the Chi^2 probability distribution.

    Broadcasting rules apply.

    Parameters
    ----------
    x : array_like or float > 0
    df : array_like or float, probably int >= 1

    Returns
    -------
    chisqprob : ndarray
        The area from `chisq` to infinity under the Chi^2 probability
        distribution with degrees of freedom `df`.

    """
    if x <= 0:
        return 1.0
    if x == 0:
        return 0.0
    if df <= 0:
        raise ValueError("Domain error.")
    if x < 1.0 or x < df:
        return 1.0 - _igam(0.5 * df, 0.5 * x)
    return _igamc(0.5 * df, 0.5 * x)


@njit('f4(f4[:], f4)', cache=True)
def polyval(p, x):
    """
    Evaluate a polynomial by Horner's scheme
    """
    y = 0
    for pv in p:
        y = y * x + pv
    return y


@njit
def ratevl(x, num, denom):  # N = M = 12
    absx = np.abs(x)
    if absx > 1:
        '''Evaluate as a polynomial in 1/x.'''
        num_ans = polyval(num[::-1], 1 / x)
        denom_ans = polyval(denom[::-1], 1 / x)
        return np.power(x, 0) * num_ans / denom_ans
    else:
        num_ans = polyval(num, x)
        denom_ans = polyval(denom, x)
    return num_ans / denom_ans


@njit
def lanczos_sum_expg_scaled(x):
    lanczos_sum_expg_scaled_num = np.array([
        0.00606184234, 0.50984166556, 19.5199278824, 449.944556906,
        6955.99960251, 75999.2930401, 601859.617168, 3481712.15498, 14605578.0876,
        43338889.3246, 86363131.2881, 103794043.116, 56906521.9134], dtype=np.float32)

    lanczos_sum_expg_scaled_denom = np.array([
        1, 66, 1925, 32670, 357423, 2637558, 13339535, 45995730,
        105258076, 150917976, 120543840, 39916800, 0], dtype=np.float32)
    return ratevl(x, lanczos_sum_expg_scaled_num, lanczos_sum_expg_scaled_denom)


@njit
def igam_fac(a, x):
    if np.abs(a - x) > 0.4 * np.abs(a):
        ax = a * np.log(x) - x - LGAMMA_05
        if (ax < -MAXLOG):
            return 0.0
        return np.exp(ax)

    fac = a + LANCZOS_G - 0.5
    res = np.sqrt(fac / np.exp(1)) / lanczos_sum_expg_scaled(a)

    if (a < 200) and (x < 200):
        res *= np.exp(a - x) * pow(x / fac, a)
    else:
        num = x - a - LANCZOS_G + 0.5
        res *= np.exp(a * (np.log1p(num / fac) - num / fac) + x * (0.5 - LANCZOS_G) / fac)
    return res


@njit
def find_inverse_gamma(a, p, q):
    """
    In order to understand what's going on here, you will
    need to refer to:

    Computation of the Incomplete Gamma Function Ratios and their Inverse
    ARMIDO R. DIDONATO and ALFRED H. MORRIS, JR.
    ACM Transactions on Mathematical Software, Vol. 12, No. 4,
    December 1986, Pages 377-393.
    """
    if a == 1:
        if q > 0.9:
            result = -np.log1p(-p)
        else:
            result = -np.log(q)
    elif a < 1:
        g = math.gamma(a)
        b = q * g

        if ((b > 0.6) or ((b >= 0.45) and (a >= 0.3))):
            '''
            DiDonato & Morris Eq 21:

            There is a slight variation from DiDonato and Morris here:
            the first form given here is unstable when p is close to 1,
            making it impossible to compute the inverse of Q(a,x) for small
            q. Fortunately the second form works perfectly well in this case.
            '''
            if (b * q > 1e-8) and (q > 1e-5):
                u = np.power(p * g * a, 1 / a)
            else:
                u = np.exp((-q / a) - EULER)
            result = u / (1 - (u / (a + 1)))
        elif (a < 0.3) and (b >= 0.35):
            '''DiDonato & Morris Eq 22:'''
            t = np.exp(-EULER - b)
            u = t * np.exp(t)
            result = t * np.exp(u)
        elif (b > 0.15) or (a >= 0.3):
            ''' DiDonato & Morris Eq 23: '''
            y = -np.log(b)
            u = y - (1 - a) * np.log(y)
            result = y - (1 - a) * np.log(u) - np.log(1 + (1 - a) / (1 + u))
        elif b > 0.1:
            ''' DiDonato & Morris Eq 24: '''
            y = -np.log(b)
            u = y - (1 - a) * np.log(y)
            result = y - (1 - a) * np.log(u) - np.log(
                (u * u + 2 * (3 - a) * u + (2 - a) * (3 - a)) / (u * u + (5 - a) * u + 2))
        else:
            ''' DiDonato & Morris Eq 25: '''
            y = -np.log(b)
            c1 = (a - 1) * np.log(y)
            c1_2 = c1 * c1
            c1_3 = c1_2 * c1
            c1_4 = c1_2 * c1_2
            a_2 = a * a
            a_3 = a_2 * a

            c2 = (a - 1) * (1 + c1)
            c3 = (a - 1) * (-(c1_2 / 2) + (a - 2) * c1 + (3 * a - 5) / 2)
            c4 = (a - 1) * ((c1_3 / 3) - (3 * a - 5) * c1_2 / 2 + (a_2 - 6 * a + 7) * c1 +
                            (11 * a_2 - 46 * a + 47) / 6)
            c5 = (a - 1) * (-(c1_4 / 4) + (11 * a - 17) * c1_3 / 6 + (-3 * a_2 + 13 * a - 13) * c1_2 +
                            (2 * a_3 - 25 * a_2 + 72 * a - 61) * c1 / 2 +
                            (25 * a_3 - 195 * a_2 + 477 * a - 379) / 12)

            y_2 = y * y
            y_3 = y_2 * y
            y_4 = y_2 * y_2
            result = y + c1 + (c2 / y) + (c3 / y_2) + (c4 / y_3) + (c5 / y_4)
    else:
        result = np.nan  # not implemented for df > 2
    return result


@njit
def igam(a, x):
    if a == 0:
        return 1 if (x > 0) else np.nan
    elif x == 0:
        '''Zero integration limit'''
        return 0
    elif np.isinf(a):
        return np.nan if np.isinf(x) else 0
    elif np.isinf(x):
        return 1
    if (x > 1.0) and (x > a):
        return 1.0 - _igamc(a, x)
    return igam_series(a, x)


@njit
def igami(a, p):
    if np.isnan(a) or np.isnan(p):
        return np.nan
    elif p == 0.0:
        return 0.0
    elif p == 1.0:
        return np.inf
    elif p > 0.9:
        return igamci(a, 1 - p)

    x = find_inverse_gamma(a, p, 1 - p)
    '''Halley's method'''
    for i in range(0, 3):
        fac = igam_fac(a, x)
        if fac == 0.0:
            return x
        f_fp = (igam(a, x) - p) * x / fac
        '''The ratio of the first and second derivatives simplifies'''
        fpp_fp = -1.0 + (a - 1) / x
        if np.isinf(fpp_fp):
            '''Resort to Newton's method in the case of overflow'''
            x = x - f_fp
        else:
            x = x - f_fp / (1.0 - 0.5 * f_fp * fpp_fp)
    return x


@njit
def igamci(a, q):
    if q == 0.0:
        return np.inf
    elif q == 1.0:
        return 0.0
    elif q > 0.9:
        return igami(a, 1 - q)

    x = find_inverse_gamma(a, 1 - q, q)
    for i in range(0, 3):
        fac = igam_fac(a, x)
        if fac == 0.0:
            return x
        f_fp = (_igamc(a, x) - q) * x / (-fac)
        fpp_fp = -1.0 + (a - 1) / x
        if np.isinf(fpp_fp):
            x = x - f_fp
        else:
            x = x - f_fp / (1.0 - 0.5 * f_fp * fpp_fp)
    return x


@njit
def chi2_isf(y, df):
    x = igamci(0.5 * df, y)
    return 2.0 * x


""" LOG-RANK CRITERIA """


# Interface functions
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
        pval = chi2_sf(chisq, df)
    except:
        pval = 1.0
    return pval
    

# def cox(durations_A, durations_B, event_observed_A=None, event_observed_B=None) -> float:
#     dfA = pd.DataFrame({'E': event_observed_A, 'T': durations_A, 'group': 0})
#     dfB = pd.DataFrame({'E': event_observed_B, 'T': durations_B, 'group': 1})
#     df_ = pd.concat([dfA, dfB])
#
#     cph = CoxPHFitter().fit(df_, 'T', 'E')
#     return cph.log_likelihood_ratio_test().p_value


# 06/02/2022 Numpy+Numba
# accelerate 2.8 according to library criteria
@njit
def iterate_coeffs_t_j(dur_A, dur_B, cens_A, cens_B, t_j, weightings):
    N_1_j = (dur_A >= t_j).sum()
    N_2_j = (dur_B >= t_j).sum()
    if N_1_j == 0 or N_2_j == 0:
        return 0, 0, 0
    O_1_j = ((dur_A == t_j) * cens_A).sum()  # np.where(dur_A == t_j, cens_A,0).sum()
    O_2_j = ((dur_B == t_j) * cens_B).sum()  # np.where(dur_B == t_j, cens_B,0).sum()
    
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
def iterate_lr_statistic(dur_A, dur_B, cens_A, cens_B, times, weightings) -> float:
    res = np.zeros((times.shape[0], 3), dtype=np.float32)
    for j, t_j in enumerate(times):
        res[j] = iterate_coeffs_t_j(dur_A, dur_B, cens_A, cens_B, t_j, weightings)

    if weightings == "peto":
        res[:, 0] = np.cumprod(res[:, 0])
    # logrank = np.dot(res[:, 0], res[:, 1])**2 / np.dot(res[:, 0]*res[:, 0], res[:, 2])
    logrank = np.power((res[:, 0]*res[:, 1]).sum(), 2) / ((res[:, 0]*res[:, 0]*res[:, 2]).sum())
    return logrank


def iterate_weight_lr_fast(dur_A, dur_B, cens_A=None, cens_B=None, weightings="") -> float:
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
        pvalue = chi2_sf(logrank, df=1)
        return pvalue
    except:
        return 1.0


# 11/08/2022 Numpy+Numba
# dur_A = np.random.choice(10000, 10000)
# cens_A = np.random.choice(2, 10000)
# dur_B = np.random.choice(10000, 10000)
# cens_B = np.random.choice(2, 10000)
# new version (for 10000 elem): 2.16 ms ± 15.2 µs per loop
# old version (for 10000 elem): 103 ms ± 171 µs per loop


# @cuda.jit
@njit('f8(f8[:], f8[:], i8[:], i8[:], i8)', cache=True)
def lr_statistic(dur_1, dur_2, cens_1, cens_2, weightings):
    times = np.unique(np.hstack((dur_1, dur_2)))
    dur_1 = np.searchsorted(times, dur_1) + 1
    dur_2 = np.searchsorted(times, dur_2) + 1
    times_range = np.array([1, times.shape[0]], dtype=np.int32)

    bins = times_range[1] - times_range[0] + 1
    n_1_j = np.histogram(dur_1, bins=bins, range=times_range)[0]
    n_2_j = np.histogram(dur_2, bins=bins, range=times_range)[0]
    O_1_j = np.histogram(dur_1 * cens_1, bins=bins, range=times_range)[0]
    O_2_j = np.histogram(dur_2 * cens_2, bins=bins, range=times_range)[0]

    N_1_j = np.cumsum(n_1_j[::-1])[::-1]
    N_2_j = np.cumsum(n_2_j[::-1])[::-1]
    ind = np.where(N_1_j * N_2_j != 0)
    N_1_j = N_1_j[ind]
    N_2_j = N_2_j[ind]
    O_1_j = O_1_j[ind]
    O_2_j = O_2_j[ind]

    N_j = N_1_j + N_2_j
    O_j = O_1_j + O_2_j
    E_1_j = N_1_j * O_j / N_j
    res = np.zeros((N_j.shape[0], 3), dtype=np.float32)
    res[:, 1] = O_1_j - E_1_j
    res[:, 2] = E_1_j * (N_j - O_j) * N_2_j / (N_j * (N_j - 1))
    res[:, 0] = 1.0
    # if np.any(N_j <= 1):
    #     return 0.0
    if weightings == 2:
        res[:, 0] = N_j
    elif weightings == 3:
        res[:, 0] = np.sqrt(N_j)
    elif weightings == 4:
        res[:, 0] = np.cumprod((1.0 - O_j / (N_j + 1)))
    logrank = np.power((res[:, 0] * res[:, 1]).sum(), 2) / ((res[:, 0] * res[:, 0] * res[:, 2]).sum())
    return logrank


def weight_lr_fast(dur_A, dur_B, cens_A=None, cens_B=None, weightings=""):
    """
    Count weighted log-rank criteria

    Parameters
    ----------
    dur_A : array-like
        Time of occurred events from first sample.
    dur_B : array-like
        Time of occurred events from second sample.
    cens_A : array-like, optional
        Indicate of occurred events from first sample.
        The default is None (all events occurred).
    cens_B : array-like, optional
        Indicate of occurred events from second sample.
        The default is None (all events occurred).
    weightings : str, optional
        Weights of criteria. The default is "" (log-rank).
        Log-rank :math:'w = 1'
        Wilcoxon :math:'w = N_j'
        Tarone-ware :math:'w = \\sqrt(N_j)'
        Peto-peto :math:'w = \\fraq{1 - O_j}{N_j + 1}'

    Returns
    -------
    logrank : float
        Chi2 statistic value of weighted log-rank test

    """
    try:
        if cens_A is None:
            cens_A = np.ones(dur_A.shape[0])
        if cens_B is None:
            cens_B = np.ones(dur_B.shape[0])
        d = {"logrank": 1, "wilcoxon": 2, "tarone-ware": 3, "peto": 4}
        weightings = d.get(weightings, 1)
        logrank = lr_statistic(dur_A.astype("float64"),
                               dur_B.astype("float64"),
                               cens_A.astype("int64"),
                               cens_B.astype("int64"),
                               np.int64(weightings))
        return logrank
    except Exception as err:
        return 0.0
