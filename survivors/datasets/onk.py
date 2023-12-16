import numpy as np
import pandas as pd
import re
import pickle 
import random
from os.path import dirname, join
from ..constants import TIME_NAME, CENS_NAME, get_y
random.seed(10)

descr_schemes = {
    '1': 'Хирургическое_лечение',
    '2': 'Лекарственная_противоопухолевая_терапия',
    '3': 'Лучевая_терапия',
    '1_1': 'Хирургическое_лечение_Первичной_опухоли_в_том_числе_с_удалением_регионарных_лимфатических_узлов',
    '1_2': 'Хирургическое_лечение_Метастазов',
    '1_3': 'Хирургическое_лечение_Симптоматическое_реконструктивно_пластическое_хирургическая_овариальная_суперссия_прочее',
    '1_4': 'Хирургическое_лечение_Выполнено_хирургическое_стадирование',
    '1_5': 'Хирургическое_лечение_Регионарных_лимфатических_узлов_без_первичной_опухоли',
    '1_6': 'Хирургическое_лечение_Криохирургия_криотерапия_лазерная_деструкция_фотодинамическая_терапия',
    '2_1': 'Лекарственная_противоопухолевая_терапия_Первая_линия',
    '2_2': 'Лекарственная_противоопухолевая_терапия_Вторая_линия',
    '2_3': 'Лекарственная_противоопухолевая_терапия_Третья_линия',
    '2_4': 'Лекарственная_противоопухолевая_терапия_Линия_после_третьей',
    '2_5': 'Лекарственная_противоопухолевая_терапия_Неоадъювантная',
    '2_6': 'Лекарственная_противоопухолевая_терапия_Адъювантная',
    '2_7': 'Лекарственная_противоопухолевая_терапия_Периоперационная_до_хирургического_лечения',
    '2_8': 'Лекарственная_противоопухолевая_терапия_Периоперационная_после_хирургического_лечения',
    '3_1': 'Лучевая_терапия_Первичной_опухоли_ложа_опухоли',
    '3_2': 'Лучевая_терапия_Метастазов',
    '3_3': 'Лучевая_терапия_Симптоматическая',
    '4': 'Химиолучевая_терапия',
    '5': 'Неспецифическое_лечение',
    '6': 'Диагностика',
}

schemes_list = list(descr_schemes.keys())
schemes_bins = ['BINS' + sch for sch in schemes_list]
schemes_nums = ['NUM' + sch for sch in schemes_list]
schemes_pred_nums = ['PRED_NUM' + sch for sch in schemes_list]
schemes_pred_bins = ['PRED_BINS' + sch for sch in schemes_list]

sign = [
    'WEI', 'HEI', 'W', 'AGE',
    'STAD', 'ONK_T', 'ONK_N', 'ONK_M', 'MTSTZ', 'INDEX_MASS',
    'PRED_SUM', 'PRED_DAYS', 'PRED_MEAN_SUM_ELEMS',
    'PRED_MEAN_DAYS_ELEMS', 'PRED_MEAN_SUM_THREADS',
    'PRED_MEAN_DAYS_THREADS', 'DELTA_PRED', 'DELTA_POST', 'PRED_THREADS',
    'DIAG'
] + schemes_bins + schemes_nums + schemes_pred_bins + schemes_pred_nums

schemes_nan_fill = schemes_bins + schemes_nums
clustering_feature = ['PROFIL', 'SMO_NAM', 'SMO', 'ST_OKATO', 'CODE_MO', 'CODE_MD', 'PRVS', 'PODR', 'NPR_MO']

categ = ['DIAG']  # ,'DS2','DS3']  # 'STAD','VID_HMP','METOD_HMP',]
    

schemes_bins = {'BINS' + i: j for i, j in descr_schemes.items()}
schemes_nums = {'NUM' + i: 'Кол_во_' + j for i, j in descr_schemes.items()}
schemes_pred_bins = {'PRED_BINS' + i: 'Пред_кол_во_' + j for i, j in descr_schemes.items()}
schemes_pred_nums = {'PRED_NUM' + i: 'Пред_лечение_' + j for i, j in descr_schemes.items()}

onk_descr = {
    'WEI': 'Вес',
    'HEI': 'Рост',
    'W': 'Пол',
    'AGE': 'Возраст',
    'STAD': 'Стадия',
    'ONK_T': 'Tumor_опухоль',
    'ONK_N': 'Nodus_узлы',
    'ONK_M': "Метастазы",
    'SUM_M': 'Стоимость_лечения_в_рублях',
    'INDEX_MASS': 'Индекс_массы',
    'PRED_SUM': 'Предыдущая_стоимость_лечения',
    'PRED_DAYS': 'Длительность_прошлой_нити',
    'PRED_MEAN_SUM_ELEMS': 'Средняя_сумма_оплаченных_позиций',
    'PRED_MEAN_DAYS_ELEMS': 'Среднее_кол_во_дней',
    'PRED_MEAN_SUM_THREADS': 'Средняя_сумма_по_нитям',
    'PRED_MEAN_DAYS_THREADS': 'Средние_дни_по_нитям',
    'DELTA_PRED': 'Дни_между_прошлым_посещением',
    'DELTA_POST': 'Дни_между_прошлой_нитью',
    'PRED_THREADS': 'Предыдущая_нить',
    'DIAG': 'Диагноз',
    'PROFIL': 'Профиль',
    'SMO_NAM': 'Название_СК',
    'SMO': 'Номер_СК',
    'ST_OKATO': 'ОКАТО',
    'CODE_MO': 'Код_больницы',
    'CODE_MD': 'Код_врача',
    'PRVS': 'Специальность_работника',
    'PODR': 'Код_отделения',
    'NPR_MO': 'Код_МО_направившей_на_лечение'
}

onk_descr.update(schemes_bins)
onk_descr.update(schemes_nums)
onk_descr.update(schemes_pred_bins)
onk_descr.update(schemes_pred_nums)

STAD_MAP_DICT = {'0': 0, 'I': 1, 'II': 2, 'III': 3, 'IV': 4}


def get_int_str(s):
    s = re.sub(r'[^0-9]', '', str(s))
    if len(s) == 0:
        return np.nan
    s = s[0]
    return int(s)


def prep_ONK(id_name, kod_name, namefile):
    ONK = pd.read_excel(namefile)
    ONK[kod_name] = ONK[kod_name].apply(lambda x: get_int_str(x))
    ONK = ONK.set_index(id_name)[kod_name].to_dict()
    return ONK


def save_pickle(obj, path):
    file_pi = open(path, 'wb') 
    pickle.dump(obj, file_pi, pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    return pickle.load(open(path, 'rb'))


def load_onk_dataset(diag={'C20', 'C50.4', 'C61'},
                     save_name="",
                     invert_death=False,  # TODO return to FALSE
                     descript=True):
    dir_env = join(dirname(__file__), "data", "ONK")
    STAD_MAP = pd.read_excel(join(dir_env, 'STAD_MAP.xlsx'))
    STAD_MAP['KOD_St'] = STAD_MAP['KOD_St'].apply(lambda x: STAD_MAP_DICT.get(re.sub(r'[^IV0]', '', x), np.nan))
    STAD_MAP = STAD_MAP.set_index('ID_St')['KOD_St'].to_dict()
    ### ONK PREP
    ONK_T = prep_ONK('ID_T', 'KOD_T', join(dir_env, 'ONK_T.xlsx'))
    ONK_N = prep_ONK('ID_N', 'KOD_N', join(dir_env, 'ONK_N.xlsx'))
    ONK_M = prep_ONK('ID_M', 'KOD_M', join(dir_env, 'ONK_M.xlsx'))
    
    sourceDF = pd.read_csv(join(dir_env, 'AGGREG_THREADS.csv'))
    if len(diag) > 0:
        sourceDF = sourceDF[sourceDF['DIAG'].isin(diag)]
        
    # if len(diag) > 0 and threadsDF[threadsDF['DIAG'] == diag].shape[0] > 0:
    #     if not os.path.exists(namespace.out +diag):
    #         os.mkdir(namespace.out +namespace.diag)
    #     namespace.out += namespace.diag + '\\'
    # #SAVE STAD
    # crl.save_pickle(STAD_MAP, namespace.out + 'STAD_MAPPING.pickle')
    # crl.save_pickle(ONK_T, namespace.out + 'ONK_T_MAPPING.pickle')
    # crl.save_pickle(ONK_N, namespace.out + 'ONK_N_MAPPING.pickle')
    # crl.save_pickle(ONK_M, namespace.out + 'ONK_M_MAPPING.pickle')
    
    sourceDF['STAD'] = sourceDF['STAD'].map(STAD_MAP)
    sourceDF['ONK_T'] = sourceDF['ONK_T'].map(ONK_T)
    sourceDF['ONK_N'] = sourceDF['ONK_N'].map(ONK_N)
    sourceDF['ONK_M'] = sourceDF['ONK_M'].map(ONK_M)
    if len(save_name) > 0:
        sourceDF.to_csv(join(dir_env, save_name), index=False)
    
    for i in categ:
        dict_encoder = {v_u: i_u for i_u, v_u in enumerate(sorted([y for y in sourceDF[i].unique() if y == y]))}
        # SAVE DICT
        #crl.save_pickle(dict_encoder, namespace.out + i + '.pickle')
        print(i, dict_encoder)
        sourceDF[i] = sourceDF[i].map(dict_encoder)
    
    if invert_death:
        sourceDF[CENS_NAME] = 1 - sourceDF['DEATH']
    else:
        sourceDF[CENS_NAME] = sourceDF['DEATH']
        
    sourceDF[TIME_NAME] = sourceDF['DAYS']
    
    ret_sign = sign
    ret_categ = categ
    ret_sch_nan = schemes_nan_fill
    if descript:
        repl = {i: onk_descr.get(i, i) for i in sign}
        sourceDF = sourceDF.rename(repl, axis=1)
        
        ret_sign = [onk_descr.get(ns, ns) for ns in ret_sign]
        ret_categ = [onk_descr.get(ns, ns) for ns in ret_categ]
        ret_sch_nan = [onk_descr.get(ns, ns) for ns in ret_sch_nan]
        
    y = get_y(cens=sourceDF[CENS_NAME], time=sourceDF[TIME_NAME])
    X = sourceDF.loc[:, ret_sign]
    return X, y, ret_sign, ret_categ, ret_sch_nan
