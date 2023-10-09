import numpy as np
import pandas as pd
import re
import pickle 
import random
from os.path import dirname, join
from ..constants import TIME_NAME, CENS_NAME, get_y
from sklearn import preprocessing

random.seed(10)

sign_gbsg = ['htreat', 'age', 'menostat', 'tumsize', 'tumgrad', 'posnodal', 'prm', 'esm']
categ_gbsg = ['htreat', 'menostat', 'tumgrad']

sign_pbc = ['trt', 'age', 'sex', 'ascites', 'hepato', 'spiders', 'edema', 'bili', 'chol',
            'albumin', 'copper', 'alk', 'ast', 'trig', 'platelet', 'protime', 'stage']
categ_pbc = ['trt', 'sex', 'ascites', 'hepato', 'spiders']


def save_pickle(obj, path):
    file_pi = open(path, 'wb') 
    pickle.dump(obj, file_pi, pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    return pickle.load(open(path, 'rb'))


def load_scheme_dataset(name):
    dir_env = join(dirname(__file__), "data")
    df = pd.read_csv(join(dir_env, f'{name}.csv'))
    df = df.rename({"time": TIME_NAME, "event": CENS_NAME}, axis=1)

    sign_c = [c for c in df.columns if c.startswith("num_") or c.startswith("fac_")]
    categ_c = [c for c in df.columns if c.startswith("fac_")]

    y = get_y(df[CENS_NAME], df[TIME_NAME].astype("int"))
    X = df.loc[:, sign_c]

    for c in categ_c:
        le = preprocessing.LabelEncoder()
        X[c] = le.fit_transform(X[c])

    if y[TIME_NAME].min() == 0:
        y[TIME_NAME] += 1
    return X, y, sign_c, categ_c, []


def load_support2_dataset():
    return load_scheme_dataset('support2')

def load_rott2_dataset():
    return load_scheme_dataset('rott2')

def load_Framingham_dataset():
    return load_scheme_dataset('Framingham')

def load_flchain_dataset():
    return load_scheme_dataset('flchain')

def load_smarto_dataset():
    return load_scheme_dataset('smarto')

def load_actg_dataset():
    return load_scheme_dataset("actg")


def load_gbsg_dataset():
    dir_env = join(dirname(__file__), "data")
    df = pd.read_csv(join(dir_env, 'GBSG.csv'))
    df = df.rename({"rfst": TIME_NAME, "cens": CENS_NAME}, axis=1)
    
    y = get_y(df[CENS_NAME], df[TIME_NAME])
    X = df.loc[:, sign_gbsg]

    if y[TIME_NAME].min() == 0:
        y[TIME_NAME] += 1
    return X, y, sign_gbsg, categ_gbsg, []


def load_pbc_dataset():
    dir_env = join(dirname(__file__), "data")
    df = pd.read_csv(join(dir_env, 'pbc.csv'))
    df = df.rename({"time": TIME_NAME, "status": CENS_NAME, "alk.phos": "alk"}, axis=1)
    df[CENS_NAME] = np.array(df[CENS_NAME] > 1, dtype=int)
    df['sex'] = df['sex'].map({'f': 1, 'm': 0})
    
    y = get_y(df[CENS_NAME], df[TIME_NAME])
    X = df.loc[:, sign_pbc]

    if y[TIME_NAME].min() == 0:
        y[TIME_NAME] += 1
    return X, y, sign_pbc, categ_pbc, []


def load_wuhan_dataset(invert_death=False):
    dir_env = join(dirname(__file__), "data")
    df = pd.read_excel(join(dir_env, 'covid_train.xlsx'))
    df['PATIENT_ID'] = df['PATIENT_ID'].fillna(method='ffill')
    columns_no_agg = ['RE_DATE', 'age', 'gender', 'Admission time', 'Discharge time', 'outcome']
    #  columns_agg = list(set(df.columns) - set(columns_no_agg))
    df_agg = df.groupby('PATIENT_ID').agg(list)
    for c in df_agg.columns:
        if c in columns_no_agg:
            df_agg[c] = df_agg[c].apply(lambda x: x[0])
        else:
            df_agg['mean_' + c] = df_agg[c].apply(np.nanmean)
            df_agg['min_' + c] = df_agg[c].apply(np.nanmin)
            df_agg['max_' + c] = df_agg[c].apply(np.nanmax)
            df_agg = df_agg.drop(c, axis=1)
    df_agg['time'] = df_agg.loc[:, ['Admission time', 'Discharge time']].apply(lambda x: (x['Discharge time'] - x['Admission time']).days, axis=1)
    df_f = df_agg[df_agg['time'] == df_agg['time']]
    df_f = df_f.drop(['RE_DATE', 'Admission time', 'Discharge time'], axis=1)
    df_f = df_f.rename({'outcome': CENS_NAME, 'time': TIME_NAME}, axis=1)
    df_f = df_f.rename({c: re.sub('[^A-Za-z0-9_]', '_', c) for c in df_f.columns}, axis=1)
    df_f = df_f.reset_index(drop=True)
    
    categ_covid = []
    sign_covid = sorted(list(set(df_f) - {CENS_NAME, TIME_NAME}))
    sign_covid = list(set(sign_covid) - {"max_2019_nCoV_nucleic_acid_detection",
                                         "mean_2019_nCoV_nucleic_acid_detection",
                                         "min_2019_nCoV_nucleic_acid_detection"})
    
    if invert_death:
        df_f[CENS_NAME] = 1 - df_f[CENS_NAME]
    y = get_y(df_f[CENS_NAME], df_f[TIME_NAME])
    X = df_f.loc[:, sign_covid]

    if y[TIME_NAME].min() == 0:
        y[TIME_NAME] += 1
    return X, y, sign_covid, categ_covid, []
