import numpy as np
from scipy.stats import norm
from .leaf_model import NonparamLeafModel, MixLeafModel, NormalizedLeafModel, MeaningLeafModel
from lifelines import KaplanMeierFitter, NelsonAalenFitter


def epanechnikov_kernel(t, T, bandwidth=1.0):
    M = 0.75 * (1 - ((t - T) / bandwidth) ** 2)
    M[abs((t - T)) >= bandwidth] = 0
    return M


class KaplanMeier:
    def __init__(self):
        self.timeline = None
        self.survival_function = None
        self.confidence_interval_ = None
        self.alpha = 0.05

    def fit(self, durations, right_censor, weights=None):
        durations = np.array(durations)
        right_censor = np.array(right_censor)
        if weights is None:
            weights = np.ones(right_censor.shape)
        self.timeline = np.unique(durations)

        dur_ = np.searchsorted(self.timeline, durations)
        hist_dur = np.bincount(dur_, weights=weights)
        self.hist_cens = np.bincount(dur_, weights=right_censor * weights)
        self.cumul_hist_dur = np.cumsum(hist_dur[::-1])[::-1]
        self.survival_function = np.hstack([1.0, np.cumprod((1.0 - self.hist_cens / (self.cumul_hist_dur)))])

    def count_confidence_interval(self):
        """ Calculated by exponential Greenwood: https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf """
        z = norm.ppf(1 - self.alpha / 2)
        cumulative_sq_ = np.sqrt(np.hstack(
            [0.0, np.cumsum(self.hist_cens / (self.cumul_hist_dur * (self.cumul_hist_dur - self.hist_cens)))]))
        np.nan_to_num(cumulative_sq_, copy=False, nan=0)
        v = np.log(self.survival_function)
        np.nan_to_num(v, copy=False, nan=0)
        self.confidence_interval_ = np.vstack([np.exp(v * np.exp(- z * cumulative_sq_ / v)),
                                               np.exp(v * np.exp(+ z * cumulative_sq_ / v))]).T
        np.nan_to_num(self.confidence_interval_, copy=False, nan=1)

    def get_confidence_interval_(self):
        if self.confidence_interval_ is None:
            self.count_confidence_interval()
        return self.confidence_interval_

    def survival_function_at_times(self, times):
        place_bin = np.digitize(times, self.timeline)
        return self.survival_function[np.clip(place_bin, 0, None)]


class FullProbKM(KaplanMeier):
    def fit(self, durations, right_censor, weights=None):
        durations = np.array(durations)
        right_censor = np.array(right_censor)
        if weights is None:
            weights = np.ones(right_censor.shape)

        self.timeline = np.unique(durations)
        right_censor = right_censor.astype("bool")
        dur_ = np.searchsorted(self.timeline, durations)

        self.hist_cens = np.bincount(dur_, weights=right_censor * weights)
        self.cumul_hist_dur = np.cumsum(self.hist_cens[::-1])[::-1]
        self.cumul_hist_dur[self.cumul_hist_dur == 0] = 1e-3  # Any cnt (in sf it becomes zero)

        self.survival_function = np.hstack([1.0, np.cumprod((1.0 - self.hist_cens / self.cumul_hist_dur))])

        N = right_censor.shape[0]
        Ncens = right_censor[~right_censor].shape[0]
        self.survival_function = Ncens / N + (1 - Ncens / N) * self.survival_function


class NelsonAalen:
    def __init__(self, smoothing=True):
        self.timeline = None
        self.survival_function = None
        self.smoothing = smoothing

    def fit(self, durations, right_censor, weights=None):
        durations = np.array(durations)
        right_censor = np.array(right_censor)
        if weights is None:
            weights = np.ones(right_censor.shape)
        # The formula Stata: https://stats.stackexchange.com/questions/6670/
        self.bandwidth = np.std(durations) / (len(durations) ** (1 / 5))
        self.timeline = np.unique(durations)

        dur_ = np.searchsorted(self.timeline, durations)
        hist_dur = np.bincount(dur_, weights=weights)
        hist_cens = np.bincount(dur_, weights=right_censor * weights)
        cumul_hist_dur = np.cumsum(hist_dur[::-1])[::-1]
        if self.smoothing and all(weights == 1):
            cumul_hist_dur = cumul_hist_dur.astype("int")
            hist_cens = hist_cens.astype("int")
            cum_ = np.cumsum(1.0 / np.arange(1, np.max(cumul_hist_dur) + 1))
            hf = cum_[cumul_hist_dur - 1] - np.where(cumul_hist_dur - hist_cens - 1 >= 0,
                                                     cum_[cumul_hist_dur - hist_cens - 1], 0)
        else:
            hf = hist_cens / cumul_hist_dur
        self.hazard_function = np.hstack([0.0, np.cumsum(hf)])

    def cumulative_hazard_at_times(self, times):
        place_bin = np.digitize(times, self.timeline)
        return self.hazard_function[np.clip(place_bin, 0, None)]

    def smoothed_hazard_(self, bandwidth):
        timeline = self.timeline
        hazard_ = np.diff(self.hazard_function)
        sh = 1.0 / bandwidth * np.dot(epanechnikov_kernel(timeline[:, None],
                                                          timeline[None, :],
                                                          bandwidth), hazard_)
        return sh + np.max(sh) / self.timeline.shape[0]

    def get_smoothed_hazard_at_times(self, bins):
        hazard_ = np.hstack([0.0, np.diff(self.cumulative_hazard_at_times(bins))])
        sh = 1.0 / self.bandwidth * np.dot(epanechnikov_kernel(bins[:, None],
                                                               bins[None, :],
                                                               self.bandwidth), hazard_)
        return sh + np.max(sh) / bins.shape[0]


class KaplanMeierZeroAfter(KaplanMeier):
    def survival_function_at_times(self, times):
        place_bin = np.searchsorted(self.timeline, times)
        # place_bin = np.digitize(times, self.timeline)  # -1
        sf = self.survival_function[np.clip(place_bin, 0, None)]
        sf[times > self.timeline[-1]] = 0
        sf[times < self.timeline[0]] = 1
        return sf


class BaseLeafModel(NonparamLeafModel):
    survival_class = KaplanMeier
    hazard_class = NelsonAalen


class BaseLeafModelOnlySurv(NonparamLeafModel):
    survival_class = KaplanMeier


class BaseLeafModelOnlyHazard(NonparamLeafModel):
    hazard_class = NelsonAalen


class BaseNormalizedLeafModel(NormalizedLeafModel):
    survival_class = KaplanMeierZeroAfter
    hazard_class = NelsonAalen


class BaseMeaningLeafModel(MeaningLeafModel):
    survival_class = KaplanMeierZeroAfter
    hazard_class = NelsonAalen


class BaseMixLeafModel(MixLeafModel):
    survival_class = KaplanMeierZeroAfter
    hazard_class = NelsonAalen


class BaseLeafModeLL(NonparamLeafModel):
    survival_class = KaplanMeierFitter
    hazard_class = NelsonAalenFitter


LEAF_NONPARAM_DICT = {
    "base": BaseLeafModel,
    "baseLL": BaseLeafModeLL,
    "only_hazard": BaseLeafModelOnlyHazard,
    "only_survive": BaseLeafModelOnlySurv,
    "base_zero_after": BaseNormalizedLeafModel,
    "base_normal": BaseNormalizedLeafModel,
    "base_meaning": BaseMeaningLeafModel,
    "base_mix": BaseMixLeafModel
}
