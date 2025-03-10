import numpy as np
import pandas as pd
import random
from os.path import dirname, join
from ..constants import TIME_NAME, CENS_NAME, get_y

random.seed(10)


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

    df = df[df[target_feat].notna().all(axis=1)].reset_index(drop=True)
    sign_c = sorted(list(set(df.columns) - set(obsolete_feat) - set(target_feat)))
    categ_c = sorted(list(set(sign_c) - set(cont_feat)))

    y = get_y(cens=df[target_feat[0]], time=df[target_feat[1]])
    X = df.loc[:, sign_c]

    if y[TIME_NAME].min() == 0:
        y[TIME_NAME] += 1
    return X, y, sign_c, categ_c, []


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
