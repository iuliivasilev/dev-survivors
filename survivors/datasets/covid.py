import numpy as np
import pandas as pd
import re
from os.path import dirname, join
from dateutil import parser  # python-dateutil
from ..constants import TIME_NAME, CENS_NAME, get_y

schemes_list = [
    'light_1', 'light_2', 'light_3',
    'middle_1', 'middle_2', 'middle_3', 'middle_4', 'middle_5', 'middle_6', 'middle_7', 'middle_8',
    'hard_1', 'hard_2', 'hard_3', 'hard_4', 'hard_5',
    'cyto_1', 'cyto_2', 'cyto_3', 'cyto_4', 'cyto_5', 'cyto_6',
    'MOS_AM', 'MOS_AM_AZ', 'N_MOS_AM_AZ', 'MOS_AM_LVF', 'N_MOS_AM_LVF',
    'MOS_CFTR', 'MOS_CFT', 'MOS_CFP', 'MOS_CFP_LVF', 'N_MOS_CFP_LVF',
    'MOS_MOK', 'MOS_PIP', 'MOS_LVF_MOK', 'mos_light_1', 'mos_light_2',
    'mos_middle_1', 'mos_middle_2', 'mos_middle_3', 'mos_middle_4',
    'mos_hard_1', 'mos_hard_2', 'mos_hard_3', 'mos_cyto_1', 'mos_cyto_2',
    # 'cheme_therap_nmg_hor_inter', 'cheme_therap_nmg_hor',
    # 'cheme_therap_nmg_inter', 'dexamethasone', 'methylprednisolone',
    # 'dex_meth'
]

important = [TIME_NAME, CENS_NAME, 'SUM_M']

categ_cyto = [
    'has_odyshka_ili_zatrudnennoe_dyhanie', 'has_oschussh_zalozhen_v_grudnoi_kletke',
    'has_slabost_ili_lomota', 'Пол', 'IGG_DEF', 'IGM_DEF', 'PCR_N',
    'сахарный диабет', 'ожирение', 'ишемическая болезнь сердца',
    'has_kashel', 'kashel_type', 'NIT', 'resultat_KT'
] + schemes_list

def_categ = {
    'osmotr_tyazhest': {'без симптомов': 0, 'легкая': 3, 'средняя': 5, 'тяжелая': 7},
    'has_odyshka_ili_zatrudnennoe_dyhanie': {'отсутствует': 0, 'присутствует': 1},
    'has_oschussh_zalozhen_v_grudnoi_kletke': {'отсутствует': 0, 'присутствует': 1},
    'has_slabost_ili_lomota': {'отсутствует': 0, 'присутствует': 1},
    'Пол': {'Мужской': 0, 'Женский': 1, 'м': 0, 'ж': 1, 'Нет данных': np.nan, 0: 0, 1: 1},
    'PCR_N': {'НЕ ОБНАРУЖЕНО': 0, 'ОБНАРУЖЕНО': 1},
    'has_kashel': {'отсутствует': 0, 'присутствует': 1},
    'kashel_type': {'сухой': 0, 'с мокротой': 1},
    'Группа риска': {'Хроники': 0, 'Беременные': 1, 'Старше 65': 2}
}

# True categorical dictionary (takes into account filled values)
def_categ_full = {
    'osmotr_tyazhest': {'без симптомов': 0, 'легкая': 3, 'средняя': 5, 'тяжелая': 7},
    'has_odyshka_ili_zatrudnennoe_dyhanie': {'отсутствует': 0, 'присутствует': 1, 0: 0, 1: 1},
    'has_oschussh_zalozhen_v_grudnoi_kletke': {'отсутствует': 0, 'присутствует': 1, 0: 0, 1: 1},
    'has_slabost_ili_lomota': {'отсутствует': 0, 'присутствует': 1, 0: 0, 1: 1},
    'Пол': {'Мужской': 0, 'Женский': 1, 'м': 0, 'ж': 1, 'Нет данных': np.nan, 0: 0, 1: 1},
    'PCR_N': {'НЕ ОБНАРУЖЕНО': 0, 'ОБНАРУЖЕНО': 1, 0: 0, 1: 1},
    'has_kashel': {'отсутствует': 0, 'присутствует': 1, 0: 0, 1: 1},
    'kashel_type': {'сухой': 0, 'с мокротой': 1, 0: 0, 1: 1},
    'Группа риска': {'Хроники': 0, 'Беременные': 1, 'Старше 65': 2, 0: 0, 1: 1, 2: 2}
}

sign = [
    # 'osmotr_tyazhest',
    'days_f', 'days_s', 'spo2', 'chdd', 'has_odyshka_ili_zatrudnennoe_dyhanie', 'temperatura_tela_value',
    'has_oschussh_zalozhen_v_grudnoi_kletke', 'has_slabost_ili_lomota', 'Пол', 'возраст',
    'IGG_N', 'IGM_N', 'IGG_DEF', 'IGM_DEF', 'PCR_N',
    'сахарный диабет', 'ожирение', 'ишемическая болезнь сердца',
    'Определение белков острой фазы С-реактивный белок',
    'has_kashel', 'kashel_type',
    'D-димер', 'Исследование ферритина',
    'NIT', 'АСТ', 'АЛТ', 'URO', 'Определение креатинина',
    'GRA#', 'Определение калия общего',
    'по Вестергрену', 'по Панченкову',
    'MON#',
    'HGB', 'MPV', 'HCT', 'PLT', 'PDW',
    'LYM#',
    'GLU', 'WBC', 'RBC',
    'RDW',
    'Абсолютное количество эозинофилов',
    'Абсолютное количество базофилов',
    'Абсолютное количество нейтрофилов',
    'resultat_KT'
] + schemes_list

therap_antic = "Гепарин Далтепарин Надропарин Эноксапарин Парнапарин Бемипарин Фондапаринукс Ривароксабан Апиксабан"
prevent_antic = therap_antic
glucoco = "Преднизолон Дексаметазон Метилпреднизолон Гидрокортизон"

mapp_sch = {
    "light_1": "Гидроксихлорохин",
    "light_2": "Мефлохин",
    "light_3": "ИФН Умифеновир",
    "middle_1": "Фавипиравир Барицитиниб Тофацитиниб",
    "middle_2": "Гидроксихлорохин Азитромицин Барицитиниб Тофацитиниб",
    "middle_3": "Мефлохин Азитромицин Барицитиниб Тофацитиниб",
    "middle_4": "Лопинавир Ритонавир ИФН Барицитиниб Тофацитиниб",
    "middle_5": "Фавипиравир Олокизумаб",
    "middle_6": "Гидроксихлорохин Азитромицин Олокизумаб",
    "middle_7": "Мефлохин Азитромицин Олокизумаб",
    "middle_8": "Лопинавир Ритонавир ИФН Олокизумаб",
    "hard_1": "Фавипиравир Тоцилизумаб Сарилумаб",
    "hard_2": "Гидроксихлорохин Азитромицин Тоцилизумаб Сарилумаб",
    "hard_3": "Мефлохин Азитромицин Тоцилизумаб Сарилумаб",
    "hard_4": "Лопинавир Ритонавир ИФН Тоцилизумаб Сарилумаб",
    "hard_5": "Лопинавир Ритонавир Гидроксихлорохин Тоцилизумаб Сарилумаб",
    "cyto_1": "Метилпреднизолон Тоцилизумаб Сарилумаб",
    "cyto_2": "Дексаметазон Тоцилизумаб Сарилумаб",
    "cyto_3": "Метилпреднизолон Канакинумаб",
    "cyto_4": "Дексаметазон Канакинумаб",
    "cyto_5": "Метилпреднизолон Дексаметазон",
    "cyto_6": "Тоцилизумаб Сарилумаб Канакинумаб",
    "MOS_AM": "Амоксициллин",
    "MOS_AM_AZ": "Ампициллин Азитромицин",     # Антибиотик: Ампициллин
    "N_MOS_AM_AZ": "Азитромицин",
    "MOS_AM_LVF": "Ампициллин Левофлоксацин",  # Антибиотик: Ампициллин
    "N_MOS_AM_LVF": "Левофлоксацин",
    "MOS_CFTR": "Цефтриаксон",                 # Антибиотик Цефтриаксон
    "MOS_CFT": "Цефотаксим",                   # Антибиотик Цефотаксим
    "MOS_CFP": "Цефепим",                      # Антибиотик Цефепим
    "MOS_CFP_LVF": "Цефепим Левофлоксацин",    # Антибиотик Цефепим
    "N_MOS_CFP_LVF": "Левофлоксацин",
    "MOS_MOK": "Моксифлоксацин",
    "MOS_PIP": "Пиперациллин",               # Антибиотик Пиперациллин

    "MOS_LVF_MOK": "Левофлоксацин Моксифлоксацин",
    'mos_light_1': "Фавипиравир Риамиловир Гидроксихлорохин ",
    'mos_light_2': "Фавипиравир Риамиловир Гидроксихлорохин " + therap_antic,
    'mos_middle_1': "Фавипиравир Риамиловир Гидроксихлорохин " + prevent_antic,
    'mos_middle_2': "Фавипиравир Риамиловир Гидроксихлорохин " + prevent_antic,
    'mos_middle_3': "Фавипиравир Риамиловир Гидроксихлорохин " + prevent_antic,
    'mos_middle_4': "Фавипиравир Риамиловир Гидроксихлорохин " + prevent_antic,
    'mos_hard_1': "Фавипиравир " + prevent_antic + glucoco,
    'mos_hard_2': "Риамиловир " + prevent_antic + glucoco,
    'mos_hard_3': "Гидроксихлорохин " + prevent_antic + glucoco,
    'mos_cyto_1': "Фондапаринукс " + glucoco,
    'mos_cyto_2': "Фондапаринукс " + prevent_antic + glucoco
}
map_bool = {'0': 'Нет', '1': 'Да'}


def create_none_ft(fill_df, sign_f, fill_none=False):
    """
    Fill nans in input DataFrame for definite features.
    Also, there is renumbering of categorical features
    (in dictionary def_categ or have a constraint for unique values).

    Parameters
    ----------
    fill_df : DataFrame
    sign_f : list
    fill_none : bool

    Returns
    -------
    DataFrame
    """
    for i in sign_f:
        # fill_df['none_'+i] = fill_df[i].apply(lambda x: int(x != x))
        if i in set(def_categ.keys()):
            # categorical features
            fill_df[i] = fill_df[i].map(def_categ[i])
            print('Categ-contin var:', i)
            if fill_none:
                fill_df[i] = fill_df[i].fillna(fill_df[i].median())
                fill_df[i] = fill_df[i].astype('int')
        else:
            # continuous features
            right_thres = 4
            if fill_none:
                right_thres = 3
                fill_df[i] = fill_df[i].fillna(fill_df[i].median())
                fill_df[i] = fill_df[i].astype('float64')
            len_uniq = len(fill_df[i].unique())
            if len_uniq < 10:
                if (len_uniq <= 6) and (len_uniq >= right_thres):
                    print('Categorical var:', i, len_uniq)
                    dict_encoder = {v_u: i_u for i_u, v_u in enumerate(np.sort(fill_df[i].unique())) if v_u == v_u}
                    fill_df[i] = fill_df[i].map(dict_encoder)
    return fill_df


def dest_date(x, y, with_neg=False):
    """
    Calculate the destination between dates.

    Parameters
    ----------
    x : str
    y : str
    with_neg : bool

    Returns
    -------
    int
    """
    days = np.nan
    try:
        p_1 = parser.parse(x)
        p_2 = parser.parse(y)
        days = (p_1 - p_2).days
        if not with_neg:
            days = abs(days)
    except Exception as e:
        pass
    return days


def load_covid_dataset(dir_env=None):
    if dir_env is None:
        dir_env = join(dirname(__file__), "data", "COVID")
    cyto = pd.read_csv(join(dir_env, "cyto_with_schemes.csv"))
    cyto = create_none_ft(cyto, sign)
    cyto[TIME_NAME] = cyto.apply(lambda row: dest_date(row['Дата создания ЭС'], row['Дата закрытия ЭС']), axis=1)
    cyto[CENS_NAME] = cyto['Причина закрытия ЭС'].apply(lambda x: int(x[0] == 'У') if x == x else 0)
    cyto['SUM_M'] = 0
    sec = cyto.loc[:, sign+important]
    
    repl = {sc: re.sub("[^0-9A-Za-zА-Яа-я_]", "", sc) for sc in sec.columns}
    sec = sec.rename(columns=repl)
    sec['idx'] = sec.index 
    sec = sec[sec[TIME_NAME] == sec[TIME_NAME]]
    for sch in schemes_list:
        sec[sch] = sec[sch].fillna(0)
    new_sign = [repl[s] for s in sign]
    new_categ = [repl[s] for s in categ_cyto]
    new_important = [repl[s] for s in schemes_list]
    
    sec = sec.reset_index()
    y = get_y(cens=sec[CENS_NAME], time=sec[TIME_NAME])
    X = sec.loc[:, new_sign]
    return X, y, new_sign, new_categ, new_important

#### ADD SCHEMES

# def add_schemes():
#     cyto = pd.read_csv(dir_env + 'cyto_train_16.csv')
#     schemes_df = pd.read_csv(dir_env + 'second_all_schemes.csv')
#     simi = pd.read_excel(dir_env + 'simi_uuid.xlsx')
#     cyto_global = cyto.merge(simi, how = 'left', on = 'uuid')
#     cyto_global['kis_uuid'] = cyto_global.apply(lambda x: x['kis_uuid'] if x['kis_uuid'] == x['kis_uuid'] else x['uuid'],axis = 1)
#     schemes_df = schemes_df.rename({'id':'kis_uuid'},axis = 1)
#     schemes_df = schemes_df.loc[:,['kis_uuid']+schemes_list]#.sum(axis = 1).value_counts()
#     cyto_global = cyto_global.merge(schemes_df, how = 'left', on = 'kis_uuid')
#     cyto_global.to_csv(dir_env + 'cyto_with_schemes.csv')
