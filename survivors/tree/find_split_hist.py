import numpy as np
from numba import njit

from scipy import stats
# from .stratified_model import KaplanMeier, FullProbKM, NelsonAalen, KaplanMeierZeroAfter
from ..external import KaplanMeier, NelsonAalen, KaplanMeierZeroAfter, FullProbKM
from ..metrics import ibs_WW, auprc
from ..constants import get_y
# import diptest

""" Auxiliary functions """


@njit('f4(f4[:], f4[:], f4[:], f4[:], u4, f4[:])', cache=True)
def lr_hist_statistic(time_hist_1, time_hist_2, cens_hist_1, cens_hist_2,
                      weightings, obs_weights):
    N_1_j = np.cumsum(time_hist_1[::-1])[::-1]
    N_2_j = np.cumsum(time_hist_2[::-1])[::-1]
    ind = np.where((cens_hist_1 + cens_hist_2 != 0) & (N_1_j * N_2_j != 0))[0]
    if ind.shape[0] == 0:
        return 0.0

    N_1_j = N_1_j[ind]  # + 1
    N_2_j = N_2_j[ind]  # + 1
    O_1_j = cens_hist_1[ind]
    O_2_j = cens_hist_2[ind]

    N_j = N_1_j + N_2_j
    O_j = O_1_j + O_2_j
    E_1_j = N_1_j * O_j / N_j
    res = np.zeros((N_j.shape[0], 3), dtype=np.float32)
    res[:, 1] = O_1_j - E_1_j  # np.abs(O_1_j - E_1_j)
    res[:, 2] = E_1_j * (N_j - O_j) * N_2_j / (N_j * (N_j))  # N_j - 1

    res[:, 0] = 1.0
    if weightings == 2:
        res[:, 0] = N_j
    elif weightings == 3:
        res[:, 0] = np.sqrt(N_j)
    elif weightings == 4:
        res[:, 0] = np.cumprod((1.0 - O_j / (N_j + 1)))
    elif weightings == 5:
        res[:, 0] = obs_weights[ind]
    elif weightings == 6:
        res[:, 0] = O_j/N_j
    elif weightings == 7:
        res[:, 0] = np.cumprod((1.0 - O_j / (N_j + 1)))
    elif weightings == 8:
        res[:, 0] = N_j/(N_1_j * N_2_j)
        # res[:, 0] = np.cumprod((1.0 - O_1_j / (N_1_j + 1))) - np.cumprod((1.0 - O_2_j / (N_2_j + 1)))
    var = (res[:, 0] * res[:, 0] * res[:, 2]).sum()
    num = res[:, 0] * res[:, 1]
    stat_val = np.power(num.sum(), 2) / var  # ((res[:, 0] * res[:, 0] * res[:, 2]).sum())

    # print("ALL:", N_j[0], "(", N_1_j[0], N_2_j[0], ")",
    #       "OBS:", np.sum(O_j), "(", O_1_j, O_2_j, ")",  # ind,
    #       "WEI:", res[:, 0],
    #       # "OBS:", np.sum(O_j), "(", np.sum(cens_hist_1), np.sum(cens_hist_2), ")",
    #       "STAT:", stat_val, "(", num, var, ")")

    if weightings == 7:
        res[:, 0] = 1 - res[:, 0]
        stat_val2 = np.power((res[:, 0] * res[:, 1]).sum(), 2) / ((res[:, 0] * res[:, 0] * res[:, 2]).sum())
        stat_val = max(stat_val, stat_val2)
    return stat_val  # It must be square of value (without sqrt)


def weight_hist_stat(time_hist_1, time_hist_2, cens_hist_1=None, cens_hist_2=None, weights_hist=None, weightings=""):
    try:
        if cens_hist_1 is None:
            cens_hist_1 = time_hist_1
        if cens_hist_2 is None:
            cens_hist_2 = time_hist_2
        if weights_hist is None:
            weights_hist = np.ones(time_hist_1.shape[0])
        d = {"logrank": 1, "wilcoxon": 2, "tarone-ware": 3, "peto": 4, "weights": 5}
        d.update({"diff": 6, "maxcombo": 7, "symm_peto": 8})
        weightings = d.get(weightings, 1)

        logrank = lr_hist_statistic(time_hist_1.astype("float32"),
                                    time_hist_2.astype("float32"),
                                    cens_hist_1.astype("float32"),
                                    cens_hist_2.astype("float32"),
                                    np.uint32(weightings),
                                    weights_hist.astype("float32")
                                    )
    except Exception as err:
        # print(err)
        # print(time_hist_1, time_hist_2, cens_hist_1, cens_hist_2, weights_hist)
        # print(time_hist_1.shape, time_hist_2.shape, cens_hist_1.shape, cens_hist_2.shape, weights_hist.shape)
        logrank = 0.0
    return logrank


# def optimal_criter_split_hist(left_time_hist, left_cens_hist,
#                               right_time_hist, right_cens_hist,
#                               na_time_hist, na_cens_hist, weights_hist, criterion, dis_coef):
#     none_to = 0
#     max_stat_val = 1.0
#
#     if int(dis_coef) > 1:
#         dis_coef = int(dis_coef)
#         left_time_hist = left_time_hist + (dis_coef - 1) * left_cens_hist
#         right_time_hist = right_time_hist + (dis_coef - 1) * right_cens_hist
#         na_time_hist = na_time_hist + (dis_coef - 1) * na_cens_hist
#
#         left_cens_hist = left_cens_hist * dis_coef
#         right_cens_hist = right_cens_hist * dis_coef
#         na_cens_hist = na_cens_hist * dis_coef
#     elif dis_coef != 0:
#         if int(1/dis_coef) > 1:
#             cens_dc = int(1/dis_coef)
#             left_time_hist = (cens_dc - 1)*left_time_hist + left_cens_hist
#             right_time_hist = (cens_dc - 1)*right_time_hist + right_cens_hist
#             na_time_hist = (cens_dc - 1)*na_time_hist + (dis_coef - 1) * na_cens_hist
#
#             left_cens_hist = left_cens_hist * cens_dc
#             right_cens_hist = right_cens_hist * cens_dc
#             na_cens_hist = na_cens_hist * cens_dc
#
#     if na_time_hist.shape[0] > 0:
#         a = weight_hist_stat(left_time_hist + na_time_hist, right_time_hist,
#                              left_cens_hist + na_cens_hist, right_cens_hist,
#                              weights_hist, weightings=criterion)
#         b = weight_hist_stat(left_time_hist, right_time_hist + na_time_hist,
#                              left_cens_hist, right_cens_hist + na_cens_hist,
#                              weights_hist, weightings=criterion)
#         # Nans move to a leaf with maximal statistical value
#         none_to = int(a < b)
#         max_stat_val = max(a, b)
#     #         print(a, b)
#     else:
#         max_stat_val = weight_hist_stat(left_time_hist, right_time_hist,
#                                         left_cens_hist, right_cens_hist,
#                                         weights_hist, weightings=criterion)
#     return (max_stat_val, none_to)


def optimal_criter_split_hist(left_time_hist, left_cens_hist,
                              right_time_hist, right_cens_hist,
                              na_time_hist, na_cens_hist, weights_hist, criterion, dis_coef,
                              apr_t_distr, apr_e_distr, l_reg):
    none_to = 0
    max_stat_val = 1.0

    # n1 = np.sum(left_time_hist)
    # n2 = np.sum(right_time_hist)
    # cf = n1 / (n1 + n2)
    cf = 0.5

    if na_time_hist.shape[0] > 0:
        a = weight_hist_stat(left_time_hist + na_time_hist + l_reg * apr_t_distr * cf,
                             right_time_hist + l_reg * apr_t_distr * (1 - cf),
                             left_cens_hist + na_cens_hist + l_reg * apr_e_distr * cf,
                             right_cens_hist + l_reg * apr_e_distr * (1 - cf),
                             weights_hist, weightings=criterion)
        b = weight_hist_stat(left_time_hist + l_reg * apr_t_distr * cf,
                             right_time_hist + na_time_hist + l_reg * apr_t_distr * (1 - cf),
                             left_cens_hist + l_reg * apr_e_distr * cf,
                             right_cens_hist + na_cens_hist + l_reg * apr_e_distr * (1 - cf),
                             weights_hist, weightings=criterion)
        # Nans move to a leaf with maximal statistical value
        none_to = int(a < b)
        max_stat_val = max(a, b)
    else:
        max_stat_val = weight_hist_stat(left_time_hist + l_reg * apr_t_distr * cf,
                                        right_time_hist + l_reg * apr_t_distr * (1 - cf),
                                        left_cens_hist + l_reg * apr_e_distr * cf,
                                        right_cens_hist + l_reg * apr_e_distr * (1 - cf),
                                        weights_hist, weightings=criterion)
    return (max_stat_val, none_to)


def split_time_to_bins(time, event=None, apr_times=None, apr_events=None):
#     if apr_times is None:
#         return np.searchsorted(np.unique(time), time)
#     return np.searchsorted(np.unique(apr_times), time)
    if apr_times is None:
        return np.searchsorted(np.arange(int(time.min() - 1), int(time.max() + 1)), time)
    return np.searchsorted(np.arange(int(apr_times.min() - 1), int(apr_times.max() + 1)), time)

    # if apr_times is None:
    #     n = np.clip(np.sum(event) // 5, 2, 20)
    #     return np.searchsorted(np.quantile(time[np.where(event)], np.arange(n + 1)/n), time)
    # n = np.clip(np.sum(apr_events) // 5, 2, 20)
    # return np.searchsorted(np.quantile(apr_times[np.where(apr_events)], np.arange(n + 1)/n), time)

#     if apr_times is None:
#         n = np.clip(time.shape[0] // 2, 2, 100)
#         return np.searchsorted(np.quantile(time, np.arange(n + 1)/n), time)
#     n = np.clip(apr_times.shape[0] // 2, 2, 100)
#     return np.searchsorted(np.quantile(apr_times, np.arange(n + 1)/n), time)

def get_attrs(max_stat_val, values, none_to, l_sh, r_sh, nan_sh, stat_diff):
    attrs = dict()
    attrs["stat_val"] = max_stat_val
    attrs["values"] = values
    attrs["stat_diff"] = stat_diff
    if none_to:
        attrs["pos_nan"] = [0, 1]
        attrs["min_split"] = min(l_sh, r_sh + nan_sh)
    else:
        attrs["pos_nan"] = [1, 0]
        attrs["min_split"] = min(l_sh + nan_sh, r_sh)
    return attrs


def transform_woe_np(x_feat, y):
    N_T = y.shape[0]
    N_D = y.sum()
    N_D_ = N_T - N_D
    x_uniq = np.unique(x_feat)
    x_dig = np.digitize(x_feat, x_uniq) - 1

    df_woe_iv = np.vstack([np.bincount(x_dig[y == 0], minlength=x_uniq.shape[0]),
                           np.bincount(x_dig[y == 1], minlength=x_uniq.shape[0])])
    all_0 = df_woe_iv[0].sum()
    all_1 = df_woe_iv[1].sum()

    p_bd = (df_woe_iv[1] + 1e-5) / (N_D + 1e-5)
    p_bd_ = (df_woe_iv[0] + 1e-5) / (N_D_ + 1e-5)
    p_b_d = (all_1 - df_woe_iv[1] + 1e-5) / (N_D + 1e-5)
    p_b_d_ = (all_0 - df_woe_iv[0] + 1e-5) / (N_D_ + 1e-5)

    woe_pl = np.log(p_bd / p_bd_)
    woe_mn = np.log(p_b_d / p_b_d_)
    descr_np = np.vstack([x_uniq, woe_pl - woe_mn])
    features_woe = dict(zip(descr_np[0], descr_np[1]))
    woe_x_feat = np.vectorize(features_woe.get)(x_feat)
    # calculate information value
    # iv = ((p_bd - p_bd_)*woe_pl).sum()
    return (woe_x_feat, descr_np)


def get_sa_hists(time, cens, minlength=1, weights=None):
    if time.shape[0] > 0:
        time_hist = np.bincount(time, minlength=minlength)
        cens_hist = np.bincount(time, weights=cens, minlength=minlength)
    else:
        time_hist, cens_hist = np.array([]), np.array([])
    return time_hist, cens_hist


def select_best_split_info(attr_dicts, type_attr, bonf=True, descr_woe=None):
    # attr_dicts = sorted(attr_dicts, key=lambda x: abs(x["stat_diff"]), reverse=True)[:max(1, len(attr_dicts)//2)]
    best_attr = max(attr_dicts, key=lambda x: x["stat_val"])

    best_attr["p_value"] = stats.chi2.sf(best_attr["stat_val"], df=1)
    best_attr["sign_split"] = len(attr_dicts)
    if best_attr["sign_split"] > 0:
        if type_attr == "cont":
            best_attr["values"] = [f" <= {best_attr['values']}", f" > {best_attr['values']}"]
        # elif type_attr == "categ":
        #     best_attr["values"] = [f" in {e}" for e in best_attr["values"]]
        elif type_attr == "woe" or type_attr == "categ":
            ind = descr_woe[1] <= best_attr["values"]
            l, r = list(descr_woe[0, ind]), list(descr_woe[0, ~ind])
            best_attr["values"] = [f" in {e}" for e in [l, r]]
        if bonf:
            best_attr["p_value"] *= best_attr["sign_split"]
    return best_attr


def ranksums_hist(x, y):
    n1 = np.sum(x)
    n2 = np.sum(y)
    rank = np.arange(1, x.shape[0] + 1)
    s = np.dot(x, rank)
    expected = n1 * (n1+n2+1) / 2.0
    z = (s - expected) / np.sqrt(n1*n2*(n1+n2+1)/12.0)
    return z

def mw_hist(x, y):
    n1 = np.sum(x)
    n2 = np.sum(y)
    rank = np.arange(1, x.shape[0] + 1)
    R1 = np.dot(x, rank)
    U1 = R1 - n1*(n1+1)/2
    U2 = n1 * n2 - U1

    U = np.maximum(U1, U2)
    return U

# def diptest_hist(x, y):
#     res = 0
#     if np.sum(x) > 0:
#         t1 = np.repeat(np.arange(1, x.size + 1), x.astype(int))
#         res = max(res, diptest.dipstat(t1))
#     if np.sum(y) > 0:
#         t2 = np.repeat(np.arange(1, y.size + 1), y.astype(int))
#         res = max(res, diptest.dipstat(t2))
#     return res

def stdtest_hist(x, y):
    res = 0
    if np.sum(x) > 0:
        t1 = np.repeat(np.arange(1, x.size + 1), x.astype(int))
        res = max(res, t1.std())
    if np.sum(y) > 0:
        t2 = np.repeat(np.arange(1, y.size + 1), y.astype(int))
        res = max(res, t2.std())
    return res


def count_sf_diff(time, cens):
    c_time = np.cumsum(time[::-1])[::-1]
    sf = np.cumprod((1.0 - cens / (c_time + 1)))
    sf[c_time == 0] = 0.0
    return np.sum((sf - 0.5)**2) / sf.shape[0]  # max(1, bins[-1] - bins[0])

def cnttest_hist(time_1, cens_1, time_2, cens_2):
    d1 = count_sf_diff(time_1, cens_1)
    d2 = count_sf_diff(time_2, cens_2)
    return min(d1, d2)


def hist_best_attr_split(arr, criterion="logrank", type_attr="cont", weights=None, thres_cont_bin_max=100,
                         signif=1.0, signif_stat=0.0, min_samples_leaf=10, bonf=True, verbose=0, balance=False,
                         apr_time=None, apr_event=None, l_reg=0, **kwargs):
    best_attr = {"stat_val": signif_stat, "p_value": signif,
                 "sign_split": 0, "values": [], "pos_nan": [1, 0]}
    if arr.shape[1] < 2 * min_samples_leaf:
        return best_attr
    vals = arr[0].astype("float")
    cens = arr[1].astype("uint")
    dur = arr[2].astype("float")
    if np.sum(cens) == 0:
        # cens = np.ones_like(dur)
        return best_attr
    if weights is None:
        weights = np.ones(dur.shape, dtype=float)  # np.ones_like(dur)
        # kmf = KaplanMeierZeroAfter()
        # kmf.fit(dur, cens)
        # dd = np.unique(dur)
        # sf = kmf.survival_function_at_times(dd)
        # sf = np.repeat(sf[np.newaxis, :], dur.shape[0], axis=0)
        #
        # y = get_y(cens=cens, time=dur)
        # weights = ibs_WW(y, y, sf, dd, axis=0)
    weights_hist = None

    if apr_time is None:
        dur = split_time_to_bins(dur, cens)
        max_bin = dur.max()
        apr_t_distr = np.zeros(max_bin + 1)
        apr_e_distr = np.zeros(max_bin + 1)
    else:
        # apr_time_1 = split_time_to_bins(apr_time, apr_time)
        apr_time_1 = split_time_to_bins(apr_time, apr_event, dur, cens)  # , apr_time)
        dur = split_time_to_bins(dur, cens)
        max_bin = apr_time_1.max()
        apr_t_distr, apr_e_distr = get_sa_hists(apr_time_1, apr_event, minlength=max_bin + 1)

    # dur = split_time_to_bins(dur)
    # max_bin = dur.max()

    ind = np.isnan(vals)

    # split nan and not-nan
    dur_notna = dur[~ind]
    cens_notna = cens[~ind]
    vals_notna = vals[~ind]
    weights_notna = weights[~ind]

    dis_coef = 1
    if balance:
        dis_coef = (cens.shape[0] - np.sum(cens)) / np.sum(cens)
        # dis_coef = max(1, (cens.shape[0] - np.sum(cens)) // np.sum(cens))

    if dur_notna.shape[0] < min_samples_leaf:
        return best_attr

    descr_woe = None
    if type_attr == "woe" or type_attr == "categ":
        vals_notna, descr_woe = transform_woe_np(vals_notna, cens_notna)

    # find splitting values
    uniq_set = np.unique(vals_notna)
    if uniq_set.shape[0] > thres_cont_bin_max:
        uniq_set = np.quantile(vals_notna, [i / float(thres_cont_bin_max) for i in range(1, thres_cont_bin_max)])
    else:
        uniq_set = (uniq_set[:-1] + uniq_set[1:]) * 0.5
    uniq_set = np.unique(np.round(uniq_set, 3))

    index_vals_bin = np.digitize(vals_notna, uniq_set, right=True)

    # find global hist by times
    na_time_hist, na_cens_hist = get_sa_hists(dur[ind], cens[ind],
                                              minlength=max_bin + 1, weights=weights[ind])
    r_time_hist, r_cens_hist = get_sa_hists(dur_notna, cens_notna,
                                            minlength=max_bin + 1, weights=weights_notna)
    l_time_hist = np.zeros_like(r_time_hist, dtype=np.float32)  # + 1
    l_cens_hist = l_time_hist.copy()

    num_nan = ind.sum()
    num_r = dur_notna.shape[0]
    num_l = 0

    if criterion == "confident" or criterion == "confident_weights":
        kmf = KaplanMeier()
        if criterion == "confident_weights":
            kmf.fit(dur, cens, weights=weights)
        else:
            kmf.fit(dur, cens)
        ci = kmf.get_confidence_interval_()
        weights_hist = 1 / (ci[1:, 1] - ci[1:, 0] + 1e-5)  # (ci[1:, 1] + ci[1:, 0] + 1e-5)
        criterion = "weights"
    elif criterion == "fullprob":
        kmf = FullProbKM()
        kmf.fit(dur, cens)
        weights_hist = kmf.survival_function_at_times(np.unique(dur))
        criterion = "weights"
    elif criterion == "ibswei":
        kmf = KaplanMeierZeroAfter()
        dur_ = arr[2].copy()
        kmf.fit(dur_, cens)

        dd = np.unique(dur_)
        sf = kmf.survival_function_at_times(dd)
        sf = np.repeat(sf[np.newaxis, :], dd.shape[0], axis=0)

        y_ = get_y(cens=np.ones_like(dd), time=dd)
        y_["cens"] = True
        ibs_ev = ibs_WW(y_, y_, sf, dd, axis=0)
        y_["cens"] = False
        ibs_cn = ibs_WW(y_, y_, sf, dd, axis=0)

        ratio = np.sum(cens)/cens.shape[0]
        weights_hist = ibs_ev*ratio + ibs_cn*(1-ratio)
        criterion = "weights"
    elif criterion == "T-ET":
        kmf = KaplanMeierZeroAfter()
        dur_ = arr[2].copy()
        kmf.fit(dur_, cens)

        dd = np.unique(dur_)
        ET = np.trapz(kmf.survival_function_at_times(dd), dd)
        weights_hist = (dd - ET)  # **2
        criterion = "weights"
    elif criterion == "kde":
        na = NelsonAalen()
        na.fit(dur, cens, np.ones(len(dur)))
        weights_hist = na.get_smoothed_hazard_at_times(np.unique(dur))
        criterion = "weights"
    # elif weights is None:
    #     weights_hist = None
    # else:
    #     weights_hist = np.bincount(dur, weights=weights,  # /sum(weights),
    #                                minlength=max_bin + 1)
    #     weights_hist /= np.bincount(dur, minlength=max_bin + 1)  # np.sqrt()
        # weights_hist = np.cumsum(weights_hist[::-1])[::-1]  # np.sqrt()

        # weights_hist = weights_hist / weights_hist.sum()

    # for each split values get branches
    attr_dicts = []
    for u in np.unique(index_vals_bin)[:-1]:
        curr_mask = index_vals_bin == u
        curr_n = curr_mask.sum()
        curr_time_hist, curr_cens_hist = get_sa_hists(dur_notna[curr_mask], cens_notna[curr_mask],
                                                      minlength=max_bin + 1, weights=weights_notna[curr_mask])
        l_time_hist += curr_time_hist
        l_cens_hist += curr_cens_hist
        r_time_hist -= curr_time_hist
        r_cens_hist -= curr_cens_hist
        num_l += curr_n
        num_r -= curr_n

        if min(num_l, num_r) <= min_samples_leaf:
            continue

        max_stat_val, none_to = optimal_criter_split_hist(
            l_time_hist, l_cens_hist, r_time_hist, r_cens_hist,
            na_time_hist, na_cens_hist, weights_hist, criterion, dis_coef,
            apr_t_distr, apr_e_distr, l_reg)

        # max_stat_val, none_to = optimal_criter_split_hist(
        #     l_time_hist, l_cens_hist, r_time_hist, r_cens_hist,
        #     na_time_hist, na_cens_hist, weights_hist, criterion, dis_coef)

        if max_stat_val > signif_stat:
            stat_diff = 1
            # if na_time_hist.shape[0] > 0:
            #     stat_diff = stdtest_hist((l_time_hist + (1 - none_to) * na_time_hist),
            #                              (r_time_hist + none_to * na_time_hist))
            # else:
            #     stat_diff = stdtest_hist(l_time_hist,
            #                              r_time_hist)

            # if na_time_hist.shape[0] > 0:
            #     stat_diff = cnttest_hist((l_time_hist + (1 - none_to) * na_time_hist),
            #                              (l_cens_hist + (1 - none_to) * na_cens_hist),
            #                              (r_time_hist + none_to * na_time_hist),
            #                              (r_cens_hist + none_to * na_cens_hist))
            # else:
            #     stat_diff = cnttest_hist(l_time_hist, l_cens_hist, r_time_hist, r_cens_hist)

            attr_loc = get_attrs(max_stat_val, uniq_set[u], none_to, num_l, num_r, num_nan, stat_diff)
            attr_dicts.append(attr_loc)

    if len(attr_dicts) == 0:
        return best_attr

    # if na_time_hist.shape[0] > 0:
    #     sh_diff = count_sf_diff(r_time_hist + l_time_hist + na_time_hist,
    #                             r_cens_hist + l_cens_hist + na_cens_hist)
    # else:
    #     sh_diff = count_sf_diff(r_time_hist + l_time_hist,
    #                             r_cens_hist + l_cens_hist)
    #
    # attr_dicts = list(filter(lambda x: x["stat_diff"] > sh_diff / 2, attr_dicts))
    # if len(attr_dicts) == 0:
    #     return best_attr

    best_attr = select_best_split_info(attr_dicts, type_attr, bonf, descr_woe=descr_woe)
    if verbose > 0:
        print(best_attr["p_value"], len(uniq_set))
    return best_attr
