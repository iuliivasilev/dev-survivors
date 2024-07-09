import pandas as pd
from ..constants import TIME_NAME, CENS_NAME, get_y
from os.path import dirname, join


def str_to_categ(df_col):
    uniq = df_col.unique()
    return df_col.map(dict(zip(uniq, range(len(uniq)))))


dir_env = join(dirname(__file__), "data", "BACKBLAZE")


def load_smart_2017_date_not_all_year():
    df = pd.read_csv(join(dir_env, 'smart_2017_date_not_all_year.csv'))
    df = df.rename({"time": TIME_NAME, "event": CENS_NAME}, axis=1)
    df = df.drop(columns=['serial_number', 'failure', 'date'])
    categ = ['model']
    df['model'] = str_to_categ(df['model'])
    sign = sorted(list(set(df.columns) - {CENS_NAME, TIME_NAME}))
    y = get_y(cens=df[CENS_NAME], time=df[TIME_NAME])
    X = df.loc[:, sign].reset_index()
    return X, y, sign, categ, []


def load_backblaze_dataset():  # load_smart_2017_date_not_in_last_report():
    df = pd.read_csv(join(dir_env, 'smart_2017_date_not_in_last_report.csv'))
    df = df.rename({"time": TIME_NAME, "event": CENS_NAME}, axis=1)
    df = df.drop(columns=['serial_number', 'failure', 'date'])
    categ = ['model']
    df['model'] = str_to_categ(df['model'])
    sign = sorted(list(set(df.columns) - {CENS_NAME, TIME_NAME}))
    y = get_y(cens=df[CENS_NAME], time=df[TIME_NAME])
    X = df.loc[:, sign].reset_index()
    return X, y, sign, categ, []


def load_smart_2017_raw_9_not_all_year():
    df = pd.read_csv(join(dir_env, 'smart_2017_raw_9_not_all_year.csv'))
    df = df.rename({"time": TIME_NAME, "event": CENS_NAME}, axis=1)
    df = df.drop(columns=['serial_number', 'failure', 'date'])
    categ = ['model']
    df['model'] = str_to_categ(df['model'])
    sign = sorted(list(set(df.columns) - {CENS_NAME, TIME_NAME}))
    y = get_y(cens=df[CENS_NAME], time=df[TIME_NAME])
    X = df.loc[:, sign].reset_index()
    return X, y, sign, categ, []


def load_smart_2017_raw_9_not_in_last_report():
    df = pd.read_csv(join(dir_env, 'smart_2017_raw_9_not_in_last_report.csv'))
    df = df.rename({"time": TIME_NAME, "event": CENS_NAME}, axis=1)
    df = df.drop(columns=['serial_number', 'failure', 'date'])
    categ = ['model']
    df['model'] = str_to_categ(df['model'])
    sign = sorted(list(set(df.columns) - {CENS_NAME, TIME_NAME}))
    y = get_y(cens=df[CENS_NAME], time=df[TIME_NAME])
    X = df.loc[:, sign].reset_index()
    return X, y, sign, categ, []
