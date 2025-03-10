import os
import random
import numpy as np
from collections import Counter
import pickle

RANDOM_STATE = 123
""" int: Fixed seed for model reproducibility """

CENS_NAME = 'cens'
TIME_NAME = 'time'
""" str: Fixed names for internal representation """

TRANSLATOR = {
    "time": {"en": "Time", "ru": "Время"},
    "sf": {"en": "Survival probability", "ru": "Вероятность выживания"},
    "hf": {"en": "Hazard rate", "ru": "Риск события"},
    "events": {"en": "events", "ru": "события"},
    "mean time": {"en": "mean time", "ru": "среднее время"},
    "size": {"en": "size", "ru": "размер"},
    "density": {"en": "density", "ru": "Плотность"},
}

def set_seed(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)


def save_pickle(obj, path):
    file_pi = open(path, 'wb')
    pickle.dump(obj, file_pi, pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    return pickle.load(open(path, 'rb'))


def get_y(cens, time, competing=False):
    """
    Internal representation of target variable

    Parameters
    ----------
    cens : array-like, shape = (n_events,)
        Censoring flags.
    time : array-like, shape = (n_events,)
        Time of occurred events.

    Returns
    -------
    y : structured array
        Output containing the binary event indicator as first field,
        and time of event or time of censoring as second field.

    """
    cens, time = np.array(cens), np.array(time) 
    y = np.empty(dtype=[(CENS_NAME, np.int32 if competing else bool),
                        (TIME_NAME, np.float64)], 
                 shape=cens.shape[0])
    y[CENS_NAME] = cens
    y[TIME_NAME] = time
    return y


def pd_to_xy(df):
    """
    Splitting pandas dataframe to feature space and target variables

    Parameters
    ----------
    df : pandas DataFrame
        Must contain columns CENS_NAME and TIME_NAME.

    Returns
    -------
    X : pandas DataFrame
        Feature space with remaining features.
    y : structured array
        y containing the binary event indicator as first field,
        and time of event or time of censoring as second field.

    """
    X = df[list(set(df.columns) - {CENS_NAME, TIME_NAME})]
    y = get_y(df[CENS_NAME], df[TIME_NAME])
    return X, y


def get_bins(time, cens=None, mode='a', num_bins=100):
    """
    Generate array of time points in timeline (from sample)

    Parameters
    ----------
    time : array-like
        Time of occurred events.
    cens : array-like, optional
        Censoring flags. The default is None (all events occurred).
    mode : str, optional
        Method of generation. The default is 'a'.
        'a' : all points (from min to max)
        'q' : quantile points (quantity is based on num_bins)
    num_bins : int, optional
        Quantity of required points. The default is 100.

    Returns
    -------
    bins : array
        Timeline

    """
    if not (cens is None):
        time = time[np.where(cens)]
    time = time.astype(np.int32)
    bins = np.array([])
    if mode == 'q':
        bins = np.quantile(time, np.arange(num_bins) / num_bins)
    elif mode == 'a':
        bins = np.arange(time.min(), time.max()+1)  # all bins
        # bins = np.unique(np.quantile(time, np.arange(2.5, 97.5) / 100))  # stable quantile bins
        # t_max = np.quantile(time, 0.95)  # NEW
        # bins = np.arange(time.min(), t_max)  # NEW
    return bins


def mode(a):
    # vals, cnts = np.unique(a, return_counts=True, equal_nan=False)
    # if vals.shape[0] == 0:
    #     return np.nan
    # modes, counts = vals[cnts.argmax()], cnts.max()
    # return modes
    return max(Counter(a).most_common(), key=lambda x: x[1] if x[0] == x[0] else 0)[0]
