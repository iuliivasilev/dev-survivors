import pandas as pd
import numpy as np

from itertools import chain, combinations
from .. import criteria as scrit 

""" Auxiliary functions """

def power_set_nonover(s):
    """
    Build a list of pairs of nonoverlapping sets
    
    Parameters
    ----------
    s : set
        Contain unique elems.

    Returns
    -------
    a : array
        Contain pairs of nonoverlapping subsets of s elements

    """
    a = np.array(list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1))))
    a = np.vstack([a[:int(a.shape[0]/2)], a[int(a.shape[0]/2):][::-1]]).T
    return a

def transform_woe(x_feat, y):
    """
    Calculate weight of evidence and transform input values.
    Also return numpy array of values mapping

    Parameters
    ----------
    x_feat : array
        Input categorical feature. Nan values ignored
    y : array
        Target binary variable.

    Returns
    -------
    woe_x_feat : array
        Mapped x_feat for woe's values.
    descr_np : array
        Array with 2 dims: source categories, mapping woe's.

    """
    a = np.vstack([x_feat,y]).T
    a = a[a[:,0] == a[:,0]]

    categs = np.unique(a[:,0]).shape[0]
    N_T = y.shape[0]
    N_D = y.sum()
    N_D_ = N_T - y.sum()

    df_woe_iv = pd.crosstab(a[:,0],a[:,1])
    all_0 = df_woe_iv[0].sum()
    all_1 = df_woe_iv[1].sum()

    df_woe_iv["p_bd"] =  (df_woe_iv[1] + 1e-5) / (N_D + 1e-5)
    df_woe_iv["p_bd_"] =  (df_woe_iv[0] + 1e-5) / (N_D_ + 1e-5)
    df_woe_iv["p_b_d"] =  (all_1 - df_woe_iv[1] + 1e-5) / (N_D + 1e-5)
    df_woe_iv["p_b_d_"] =  (all_0 - df_woe_iv[0] + 1e-5) / (N_D_ + 1e-5)

    df_woe_iv["woe_pl"]= np.log(df_woe_iv["p_bd"] / df_woe_iv["p_bd_"])
    df_woe_iv["woe_mn"]= np.log(df_woe_iv["p_b_d"] / df_woe_iv["p_b_d_"])
    features_woe = (df_woe_iv["woe_pl"] - df_woe_iv["woe_mn"]).to_dict()
    descr_np = np.vstack([df_woe_iv.index, (df_woe_iv["woe_pl"] - df_woe_iv["woe_mn"])])
    woe_x_feat = np.vectorize(features_woe.get)(x_feat)
    # iv = ((df_woe_iv["p_bd"].to_numpy() - df_woe_iv["p_bd_"].to_numpy())*df_woe_iv["woe_pl"].to_numpy()).sum() # calculate information value
    return (woe_x_feat, descr_np)

def optimal_criter_split(arr_nan, left, right, criterion):
    """
    Define best split according to criterion (depends on nan allocation).
    Nan values add into turn branches (left and right), count all p-values.
    Choose minimal p-value and nan's allocation

    Parameters
    ----------
    arr_nan : structured array
        Contain censuring flag and time of events with feature missing values.
    left : structured array
        Contain censuring flag and time of left sample events.
    right : structured array
        Contain censuring flag and time of right sample events.
    criterion : function
        Weighted log-rank criteria.

    Returns
    -------
    min_p_val : float
        P-value of best split.
    none_to : int
        Number of branch for allocation (default 0)
        0 : to left branch
        1 : to right branch

    """
    none_to = 0
    min_p_val = 1.0
    if arr_nan.shape[1] > 0:
        left_and_nan = np.hstack([left, arr_nan])
        right_and_nan = np.hstack([right, arr_nan])
        a = criterion(left_and_nan[1], right[1], left_and_nan[0], right[0])
        b = criterion(left[1], right_and_nan[1], left[0], right_and_nan[0])
        # Nans move to a leaf with less p-value
        none_to = int(a > b)
        min_p_val = min(a,b)
    else:
        min_p_val = criterion(left[1], right[1], left[0], right[0])
    return (min_p_val, none_to)

def get_attrs(min_p_value, values, none_to, l_sh, r_sh, nan_sh):
    """
    Create dictionary of best split.

    Parameters
    ----------
    min_p_value : float
        P-value of best split.
    values : object
        Splitting value (can be single number or list).
    none_to : int
        Number of branch for allocation.
    l_sh : int
        Size of left branch.
    r_sh : int
        Size of right branch.
    nan_sh : int
        Quantitive of nan events.

    Returns
    -------
    attrs : dict
        Contain fields about best split.
        p_value : minimal p-value
        values : splitting value
        pos_nan : list of binary values of allocation nans
        min_split : minimal size of branch

    """
    attrs = {}
    attrs["p_value"] = min_p_value
    attrs["values"] = values
    if none_to:
        attrs["pos_nan"] = [0,1]
        attrs["min_split"] = min(l_sh,r_sh+nan_sh)
    else:
        attrs["pos_nan"] = [1,0]
        attrs["min_split"] = min(l_sh+nan_sh,r_sh)
    return attrs

def get_cont_attrs(uniq_set, arr_notnan, arr_nan, min_samples_leaf, criterion, 
                   signif, thres_cont_bin_max):
    """
    Find best split for continious feature.
    Consider all intermiate points of values (with quantile discretization).
    For each points define two branches and count p-value.
    Insignificant and too small splits aren't considered. 

    Parameters
    ----------
    uniq_set : set
        Unique values of feature.
    arr_notnan : structured array
        Contain censuring flag and time of events with feature missing values.
    arr_nan : structured array
        Contain feature value, censuring flag and time of events.
    min_samples_leaf : int
        Minimal acceptable size of branches.
    criterion : function
        Weighted log-rank criteria.
    signif : float
        Minimal acceptable significance of split.
    thres_cont_bin_max : int
        Maximal quantitive of intermiate points.

    Returns
    -------
    attr_dicts : list
        List of attr dicts (contain fields about best split).

    """
    if uniq_set.shape[0] > thres_cont_bin_max:
            uniq_set = np.quantile(arr_notnan[0], 
                    [i/float(thres_cont_bin_max) for i in range(1,thres_cont_bin_max)])
    else: # Set intermediate points
        uniq_set = (uniq_set[:-1] + uniq_set[1:])*0.5
    uniq_set = np.round(uniq_set, 3)
    attr_dicts = []
    for value in uniq_set:
        # Filter by attr value
        ind = arr_notnan[0]>=value
        left = arr_notnan[1:, np.where(ind)[0]].astype(np.int32)
        right = arr_notnan[1:, np.where(~ind)[0]].astype(np.int32)
        if min(left.shape[1],right.shape[1]) <= min_samples_leaf:
            continue
        min_p_value, none_to = optimal_criter_split(arr_nan, left, right, criterion)
        if min_p_value <= signif:
            attr_loc = get_attrs(min_p_value, value, none_to, 
                                 left.shape[1], right.shape[1], arr_nan.shape[1])
            attr_dicts.append(attr_loc)
    return attr_dicts

def get_categ_attrs(uniq_set, arr_notnan, arr_nan, min_samples_leaf, criterion, signif):
    """
    Find best split for categorical feature.
    Consider all nonoverlapping subsets of uniq elements.
    For each subset define branche and count p-value.
    Insignificant and too small splits aren't considered. 

    Parameters
    ----------
    uniq_set : set
        Unique values of feature.
    arr_notnan : structured array
        Contain censuring flag and time of events with feature missing values.
    arr_nan : structured array
        Contain feature value, censuring flag and time of events.
    min_samples_leaf : int
        Minimal acceptable size of branches.
    criterion : function
        Weighted log-rank criteria.
    signif : float
        Minimal acceptable significance of split.

    Returns
    -------
    attr_dicts : list
        List of attr dicts (contain fields about best split).

    """
    attr_dicts = []
    pairs_uniq = power_set_nonover(uniq_set)
    for l,r in pairs_uniq:
        left = arr_notnan[1:, np.isin(arr_notnan[0], l)].astype(np.int32)
        right = arr_notnan[1:, np.isin(arr_notnan[0], r)].astype(np.int32)
        if min(left.shape[1],right.shape[1]) <= min_samples_leaf:
            continue
        min_p_value, none_to = optimal_criter_split(arr_nan, left, right, criterion)
        if min_p_value <= signif:
            attr_loc = get_attrs(min_p_value, [list(l),list(r)], none_to, 
                                 left.shape[1], right.shape[1], arr_nan.shape[1])
            attr_dicts.append(attr_loc)
    return attr_dicts

def best_attr_split(arr, criterion = "logrank", type_attr = "cont", thres_cont_bin_max = 100,
              signif = 1.0, min_samples_leaf = 10, bonf = True, verbose = 0, 
              **kwargs):
    """
    Choose best split for fixed feature.
    Find best splits and choose partition with minimal p-value.

    Parameters
    ----------
    arr : structured array
        Contain feature value, censuring flag and time of events.
    criterion : object, optional
        Function or name of criteria. The default is "logrank".
        If str then replace to realization from criteria_dict
    type_attr : str, optional
        Continious, categorical or WOE features. The default is "cont".
        Continious : get_cont_attrs
        Categorical : get_categ_attrs
        WOE : mapping from categorical to continious and apply get_cont_attrs
    thres_cont_bin_max : int, optional
        Maximal quantitive of intermiate points.
    signif : TYPE, optional
        Minimal acceptable significance of split. The default is 1.0.
    min_samples_leaf : TYPE, optional
        Minimal acceptable size of branches. The default is 10.
    bonf : boolean, optional
        Apply Bonferroni adjustment. The default is True.
    verbose : int, optional
        Print best splits if verbose > 0. The default is 0.
    **kwargs : None
        Undefined parameters.

    Returns
    -------
    best_attr : dict
        Contain fields about best split.
        p_value : minimal p-value
        values : list of splitting rules
        pos_nan : list of binary values of allocation nans
        min_split : minimal size of branch
        sign_split : quantitive of significant splits

    """
    if criterion in scrit.CRITERIA_DICT:
        criterion = scrit.CRITERIA_DICT[criterion]
        
    best_attr = {"p_value":signif, "sign_split" : 0,
                  "values":[], "pos_nan":[1,0]}
    attr_dicts = [best_attr]
    # The leaf is too small for split
    if arr.shape[1] < 2*min_samples_leaf:
        return best_attr 
    ind = np.isnan(arr[0])
    arr_nan = arr[1:, np.where(ind)[0]].astype(np.int32)
    arr_notnan = arr[:, np.where(~ind)[0]]
    
    if type_attr == "woe":
        arr_notnan[0], descr_np = transform_woe(arr_notnan[0], arr_notnan[1])
    uniq_set = np.unique(arr_notnan[0])
    
    if type_attr == "categ" and uniq_set.shape[0] > 0:
        attr_dicts = get_categ_attrs(uniq_set, arr_notnan, arr_nan, min_samples_leaf, criterion, signif)
    else:
        attr_dicts = get_cont_attrs(uniq_set, arr_notnan, arr_nan, min_samples_leaf, criterion, signif, thres_cont_bin_max)
    
    if len(attr_dicts) == 0:
        return best_attr
    best_attr = min(attr_dicts,key=lambda x:x["p_value"])
    best_attr["sign_split"] = len(attr_dicts)
    if best_attr["sign_split"] > 0:
        if type_attr == "cont":
            best_attr["values"] = [' >= %s' % (best_attr["values"]), 
                                   ' < %s' % (best_attr["values"])]
        elif type_attr == "categ":
            best_attr["values"] = [' in %s' % (e) for e in best_attr["values"]]
        elif type_attr == "woe":
            ind = descr_np[1] >= best_attr["values"]
            l,r = list(descr_np[0, np.where(ind)[0]]), list(descr_np[0, np.where(~ind)[0]])
            best_attr["values"] = [' in %s' % (e) for e in [l,r]]
        # Bonferroni adjustment
        if bonf:
            best_attr["p_value"] *= best_attr["sign_split"]
        if verbose > 0:
            print(best_attr["p_value"], len(uniq_set))
    return best_attr