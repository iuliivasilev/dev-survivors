import numpy as np
import pandas as pd
import random
from os.path import dirname, join
from ..constants import TIME_NAME, CENS_NAME, get_y

random.seed(10)


def prepare_dataset_by_template(df, obsolete_feat, target_feat, cont_feat, competing=False):
    df = df[df[target_feat].notna().all(axis=1)].reset_index(drop=True)
    sign_c = sorted(list(set(df.columns) - set(obsolete_feat) - set(target_feat)))
    categ_c = sorted(list(set(sign_c) - set(cont_feat)))

    y = get_y(cens=df[target_feat[0]], time=df[target_feat[1]], competing=competing)
    X = df.loc[:, sign_c]

    if y[TIME_NAME].min() == 0:
        y[TIME_NAME] += 1
    return X, y, sign_c, categ_c, []


def load_ecomm_dataset():
    """
    https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
    """
    dir_env = join(dirname(__file__), "data", "CRM")
    df = pd.read_excel(join(dir_env, 'ECommerce.xlsx'), sheet_name="E Comm")

    obsolete_feat = ["CustomerID"]
    target_feat = ["Churn", "DaySinceLastOrder"]
    cont_feat = ["Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp", "NumberOfDeviceRegistered",
                 "SatisfactionScore", "NumberOfAddress", "Complain", "OrderAmountHikeFromlastYear",
                 "CouponUsed", "OrderCount", "CashbackAmount"]
    return prepare_dataset_by_template(df, obsolete_feat, target_feat, cont_feat)


def load_telco_dataset():
    """
    https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction/data
    https://www.interviewquery.com/p/customer-churn-datasets
    https://github.com/Pradnya1208/Telecom-Customer-Churn-prediction
    https://github.com/archd3sai/Customer-Survival-Analysis-and-Churn-Prediction/blob/master/Customers%20Survival%20Analysis.ipynb
    """
    pass
    return None


def load_bank_dataset():
    """
    https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
    """
    pass
    return None


def load_cell2cell_dataset(competing=False):
    """
    https://www.kaggle.com/datasets/jpacse/telecom-churn-new-cell2cell-dataset
    https://github.com/jmoy409/The_Churn_Game/tree/master
    https://github.com/lmutesi/Cell2Cell-Churn-Analysis/tree/main
    https://www.interviewquery.com/p/customer-churn-datasets#advanced-customer-churn-datasets-projects
    """
    dir_env = join(dirname(__file__), "data", "CRM")
    df = pd.read_csv(join(dir_env, 'cell2cell-duke univeristy.csv.gz'), compression='gzip')

    obsolete_feat = ["customer", "traintest", "churndep", "eqpdays", "changem", "changer", "retcalls",
                     "retaccpt", "refer", "incmiss", "income", "mcycle", "setprcm", "setprc", "retcall"]
    target_feat = ["churn", "months"]
    cont_feat = sorted(list(set(df.select_dtypes(include=np.number).columns) - set(obsolete_feat) - set(target_feat)))
    if competing:
        """
        0    49150 - no churn, no new offers (renew)
        1     1288 - no churn, transfer to new offers
        2    19479 - churn, no new offers
        3     1130 - churn, resumed subscription after a while
        """
        df["churn"] = df["churn"] * 2 + df["retcall"]
    return prepare_dataset_by_template(df, obsolete_feat, target_feat, cont_feat, competing)


def load_gym_dataset():
    """
    https://www.kaggle.com/datasets/adrianvinueza/gym-customers-features-and-churn/data
    https://www.interviewquery.com/p/customer-churn-datasets#advanced-customer-churn-datasets-projects
    """
    pass
    return None
