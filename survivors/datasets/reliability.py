import pandas as pd
from os.path import dirname, join
from ..constants import TIME_NAME, get_y


def prepare_dataset_by_template(df, obsolete_feat, target_feat, cont_feat, competing=False):
    df = df[df[target_feat].notna().all(axis=1)].reset_index(drop=True)
    sign_c = sorted(list(set(df.columns) - set(obsolete_feat) - set(target_feat)))
    categ_c = sorted(list(set(sign_c) - set(cont_feat)))

    y = get_y(cens=df[target_feat[0]], time=df[target_feat[1]], competing=competing)
    X = df.loc[:, sign_c]

    if y[TIME_NAME].min() == 0:
        y[TIME_NAME] += 1
    return X, y, sign_c, categ_c, []


def load_alibaba_dataset(threshold=0.99, without_meta=True):
    """
    https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
    """
    dir_env = join(dirname(__file__), "data", "RELIABILITY")
    df = pd.read_csv(join(dir_env, 'Cut_Alibaba.csv.gz'), compression='gzip')
    if without_meta:
        df = df.iloc[:, 3:]
    target_feat = ["event", "event_time"]
    nan_percentage = df.isna().mean()
    columns_with_high_nan = nan_percentage[nan_percentage >= threshold].index.tolist()
    cont_feat = df.columns
    return prepare_dataset_by_template(df, columns_with_high_nan, target_feat, cont_feat)
