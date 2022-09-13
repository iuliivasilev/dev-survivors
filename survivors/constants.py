import os
import random
import numpy as np

RANDOM_STATE = 123
""" int: Fixed seed for model reproducibility """

CENS_NAME = 'cens'
TIME_NAME = 'time'
""" str: Fixed names for internal representation """


def set_seed(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)


def get_y(cens, time):
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
    y = np.empty(dtype=[(CENS_NAME, bool), 
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
        Must to have columns CENS_NAME and TIME_NAME.

    Returns
    -------
    X : pandas DataFrame
        Feature space with remaining features.
    y : structured array
        y containing the binary event indicator as first field,
        and time of event or time of censoring as second field.

    """
    X = df.loc[:, list(set(df.columns) - {CENS_NAME, TIME_NAME})]
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
        Array of time points in timeline

    """
    if not(cens is None):
        time = time[np.where(cens)]
    time = time.astype(np.int32)
    bins = np.array([])
    if mode == 'q':
        bins = np.quantile(time, np.arange(num_bins) / num_bins)
    elif mode == 'a':
        bins = np.arange(time.min(), time.max()+1)
    return bins
