from .other import load_gbsg_dataset
from .other import load_pbc_dataset
from .other import load_wuhan_dataset

from .other import load_actg_dataset
from .other import load_flchain_dataset
from .other import load_smarto_dataset
from .other import load_rott2_dataset
from .other import load_support2_dataset
from .other import load_Framingham_dataset
from .other import load_metabric_dataset

from .covid import load_covid_dataset
from .onk import load_onk_dataset
from .backblaze import load_backblaze_dataset

from .new_backblaze import load_backblaze_2016_2018
from .new_backblaze import load_backblaze_2018_2021
from .new_backblaze import load_backblaze_2021_2023

DATASETS_LOAD = {
    "GBSG": load_gbsg_dataset,
    "PBC": load_pbc_dataset,
    "WUHAN": load_wuhan_dataset,
    "actg": load_actg_dataset,
    "flchain": load_flchain_dataset,
    "smarto": load_smarto_dataset,
    "rott2": load_rott2_dataset,
    "support2": load_support2_dataset,
    "Framingham": load_Framingham_dataset,
    "backblaze16_18": load_backblaze_2016_2018,
    "backblaze18_21": load_backblaze_2018_2021,
    "backblaze21_23": load_backblaze_2021_2023,
    "metabric": load_metabric_dataset,
    # "backblaze": load_backblaze_dataset
    # "ONK": load_onk_dataset,
    # "COVID": load_covid_dataset,
}
