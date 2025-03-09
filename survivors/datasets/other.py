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
    """
    Base loader function for internal datasets

    Parameters
    ----------
    name: str

    Returns
    -------
    X: pd.DataFrame
        Feature space
    y: structured np.ndarray
        Target variables
    sign: list
        All variables
    categ: list
        Categorical variables
    sch_nan: list
        Scheme variables (supported multiple prediction)

    """
    dir_env = join(dirname(__file__), "data")
    df = pd.read_csv(join(dir_env, f'{name}.csv'))
    df = df.rename({"time": TIME_NAME, "event": CENS_NAME}, axis=1)

    sign_c = [c for c in df.columns if c.startswith("num_") or c.startswith("fac_")]
    categ_c = [c for c in df.columns if c.startswith("fac_")]

    y = get_y(cens=df[CENS_NAME], time=df[TIME_NAME].astype("int"))
    X = df.loc[:, sign_c]

    for c in categ_c:
        le = preprocessing.LabelEncoder()
        X[c] = le.fit_transform(X[c])

    if y[TIME_NAME].min() == 0:
        y[TIME_NAME] += 1
    return X, y, sign_c, categ_c, []


def load_support2_dataset():
    """
    Full description: https://archive.ics.uci.edu/dataset/880/support2
    """
    return load_scheme_dataset('support2')

def load_rott2_dataset():
    """
    Full description: https://rdrr.io/cran/AF/man/rott2.html
    """
    return load_scheme_dataset('rott2')

def load_Framingham_dataset():
    """
    Full description: https://search.r-project.org/CRAN/refmans/riskCommunicator/html/framingham.html
    """
    return load_scheme_dataset('Framingham')

def load_flchain_dataset():
    """
    Full description: https://pmagunia.com/dataset/r-dataset-package-survival-flchain
    """
    return load_scheme_dataset('flchain')

def load_smarto_dataset():
    """
    Full description: https://pubmed.ncbi.nlm.nih.gov/36806141/
    """
    return load_scheme_dataset('smarto')

def load_actg_dataset():
    """
    Full description: https://rdrr.io/cran/mlr3proba/man/actg.html
    """
    return load_scheme_dataset("actg")


def load_gbsg_dataset():
    """
    Full description: https://paperswithcode.com/dataset/gbsg2
    """
    dir_env = join(dirname(__file__), "data")
    df = pd.read_csv(join(dir_env, 'GBSG.csv'))
    df = df.rename({"rfst": TIME_NAME, "cens": CENS_NAME}, axis=1)
    
    y = get_y(cens=df[CENS_NAME], time=df[TIME_NAME])
    X = df.loc[:, sign_gbsg]

    if y[TIME_NAME].min() == 0:
        y[TIME_NAME] += 1
    return X, y, sign_gbsg, categ_gbsg, []


def load_pbc_dataset():
    """
    Full description: https://stat.ethz.ch/R-manual/R-patched/library/survival/html/pbc.html
    """
    dir_env = join(dirname(__file__), "data")
    df = pd.read_csv(join(dir_env, 'pbc.csv'))
    df = df.rename({"time": TIME_NAME, "status": CENS_NAME, "alk.phos": "alk"}, axis=1)
    df[CENS_NAME] = np.array(df[CENS_NAME] > 1, dtype=int)
    df['sex'] = df['sex'].map({'f': 1, 'm': 0})
    
    y = get_y(cens=df[CENS_NAME], time=df[TIME_NAME])
    X = df.loc[:, sign_pbc]

    if y[TIME_NAME].min() == 0:
        y[TIME_NAME] += 1
    return X, y, sign_pbc, categ_pbc, []


def load_metabric_dataset():
    """
    https://github.com/baskayj/Censoring-sensitivity-analysis-for-survival-models/tree/main
    """
    dir_env = join(dirname(__file__), "data")
    df = pd.read_csv(join(dir_env, f'metabric.tsv'), sep="\t", header=4)
    obsolete_feat = ["PATIENT_ID", "VITAL_STATUS", "RFS_MONTHS", "RFS_STATUS",
                     "BREAST_SURGERY", "SEX", "LATERALITY", "ER_IHC"]
    target_feat = ["OS_STATUS", "OS_MONTHS"]
    cont_feat = ["LYMPH_NODES_EXAMINED_POSITIVE", "NPI", "CELLULARITY", "AGE_AT_DIAGNOSIS"]

    df = df[df[target_feat].notna().all(axis=1)].reset_index(drop=True)

    df["CELLULARITY"] = df["CELLULARITY"].map({"Low": 0, "Moderate": 1, "High": 2})
    df["CHEMOTHERAPY"] = df["CHEMOTHERAPY"].map({"YES": True, "NO": False})
    df["ER_IHC_POS"] = df["ER_IHC"].map({"Positve": True, "Negative": False})
    df["HORMONE_THERAPY"] = df["HORMONE_THERAPY"].map({"YES": True, "NO": False})
    df["RADIO_THERAPY"] = df["RADIO_THERAPY"].map({"YES": True, "NO": False})
    df["BREAST_SURGERY_MASTECTOMY"] = df["BREAST_SURGERY"].map({"MASTECTOMY": True, "BREAST CONSERVING": False})
    df["LATERALITY_LEFT"] = df["LATERALITY"].map({"Left": True, "Right": False})
    df["RFS_RECURRED"] = df["RFS_STATUS"].map({"1:Recurred": True, "0:Not Recurred": False})
    df["INFERRED_MENOPAUSAL_POST"] = df["INFERRED_MENOPAUSAL_STATE"].map({"Post": True, "Pre": False})

    sign_c = sorted(list(set(df.columns) - set(obsolete_feat) - set(target_feat)))
    categ_c = sorted(list(set(sign_c) - set(cont_feat)))

    y = get_y(cens=df["OS_STATUS"] == "1:DECEASED", time=(round(df["OS_MONTHS"])).astype(int))
    X = df.loc[:, sign_c]

    if y[TIME_NAME].min() == 0:
        y[TIME_NAME] += 1
    return X, y, sign_c, categ_c, []


def load_seer_dataset():
    """
    https://github.com/thecml/baysurv
    """
    dir_env = join(dirname(__file__), "data")
    df = pd.read_csv(join(dir_env, f'seer.csv'))
    obsolete_feat = ["A_Stage"]
    target_feat = ["Status", "Survival_Months"]
    cont_feat = ["Age", "Race", "Marital_Status", "T_Stage", "N_Stage", "6th_Stage",
                 "differentiate", "Grade", "A_Stage_Distant", "Tumor_Size", "Estrogen_Status",
                 "Progesterone_Status", "Regional_Node_Examined", "Reginol_Node_Positive"]
    df["Status"] = df["Status"].map({"Alive": 0, "Dead": 1})
    df["T_Stage"] = df["T_Stage"].str[1:].astype(int)
    df["N_Stage"] = df["N_Stage"].str[1:].astype(int)
    df["A_Stage_Distant"] = df["A_Stage"].map({"Distant": True, "Regional": False})
    df["Estrogen_Status"] = df["Estrogen_Status"].map({"Positive": True, "Negative": False})
    df["Progesterone_Status"] = df["Progesterone_Status"].map({"Positive": True, "Negative": False})

    df = df[df[target_feat].notna().all(axis=1)].reset_index(drop=True)
    sign_c = sorted(list(set(df.columns) - set(obsolete_feat) - set(target_feat)))
    categ_c = sorted(list(set(sign_c) - set(cont_feat)))

    y = get_y(cens=df["Status"], time=df["Survival_Months"])
    X = df.loc[:, sign_c]

    if y[TIME_NAME].min() == 0:
        y[TIME_NAME] += 1
    return X, y, sign_c, categ_c, []


def load_mimic_dataset():
    """
    https://github.com/thecml/baysurv
    """
    dir_env = join(dirname(__file__), "data")
    df = pd.read_csv(join(dir_env, f'mimic.csv.gz'), compression='gzip')
    sign_c = sorted(list(set(df.columns) - set(["time", "event"])))

    y = get_y(cens=df["event"], time=df["time"])
    X = df.loc[:, sign_c]
    if y[TIME_NAME].min() == 0:
        y[TIME_NAME] += 1
    return X, y, sign_c, [], []


def load_wuhan_dataset(invert_death=False):
    """
    Full description: https://www.nature.com/articles/s42256-020-0180-7
    """
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
    y = get_y(cens=df_f[CENS_NAME], time=df_f[TIME_NAME])
    X = df_f.loc[:, sorted(sign_covid)]

    if y[TIME_NAME].min() == 0:
        y[TIME_NAME] += 1
    return X, y, sign_covid, categ_covid, []
