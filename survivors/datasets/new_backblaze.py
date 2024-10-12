import pandas as pd
from ..constants import TIME_NAME, CENS_NAME, get_y
from os.path import dirname, join


dir_env = join(dirname(__file__), "data", "BACKBLAZE")


def str_to_categ(df_col):
    uniq = df_col.unique()
    return df_col.map(dict(zip(uniq, range(len(uniq)))))


def load_backblaze(file_name, threshold=0.99):
    df = pd.read_csv(join(dir_env, file_name))
    df['time'] = pd.to_timedelta(df['time']).dt.days
    df = df.rename({"time": TIME_NAME, "event": CENS_NAME}, axis=1)
    nan_percentage = df.isna().mean()
    columns_with_high_nan = nan_percentage[nan_percentage >= threshold].index.tolist()
    df = df.drop(columns=['serial_number', 'date', 'time_row'] + columns_with_high_nan)
    categ = ['model']
    df['model'] = str_to_categ(df['model'])
    sign = sorted(list(set(df.columns) - {CENS_NAME, TIME_NAME}))
    y = get_y(df[CENS_NAME], df[TIME_NAME] + 1)
    X = df.loc[:, sign]
    return X, y, sign, categ, []


def load_backblaze_2016_2018(threshold=0.99):
    return load_backblaze('backblaze_drop_truncated_2016_2018.csv', threshold=0.99)


def load_backblaze_2018_2021(threshold=0.99):
    return load_backblaze('backblaze_drop_truncated_2018_2021.csv', threshold=0.99)


def load_backblaze_2021_2023(threshold=0.99):
    return load_backblaze('backblaze_drop_truncated_2021_2023.csv', threshold=0.99)
