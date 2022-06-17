import pandas as pd
import numpy as np
import os
import time
import copy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from graphviz import Digraph
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis

from .. import metrics as metr
from .node import Node
from .. import constants as cnt


def format_to_pandas(X, columns):
    type_df = type(X)
    if type_df.__name__ == "DataFrame":
        return X.loc[:,columns]
    elif type_df.__name__ == "ndarray":
        return pd.DataFrame(X, columns = columns)
    return None

""" Functions of prunning """

def ols(a,b):
    return sum((a - b)**2)

def get_full_copy(tree):
    return copy.deepcopy(tree)

def count_leaf(tree):
    if tree.is_leaf:
        return np.array([tree.numb])
    arr = np.array([])
    for edge in tree.edges:
        arr = np.append(arr, count_leaf(edge))
    return arr.flatten().astype('int')

def count_spanning_leaf(tree):
    if tree.is_leaf:
        return np.array([])
    arr = np.array([])
    for edge in tree.edges:
        arr = np.append(arr, count_spanning_leaf(edge))
    if arr.shape[0] == 0:
        return np.array([tree.numb])
    return arr.flatten().astype('int')

def delete_sub_leaf(tree, leaf_list):
    if tree.numb in leaf_list:
        tree.set_leaf()
    for edge in tree.edges:
        delete_sub_leaf(edge, leaf_list)

def find_best_uncut(tree, X, y, target, mode_f, choose_f):
    span_leaf = count_spanning_leaf(tree)
    # print(span_leaf)
    d = {}
    for el in span_leaf:
        y_pred = tree.predict(X, target = target, end_list = [el])
        d[el] = round(mode_f(y,y_pred),4)
    
    new_leaf, val = choose_f(d.items(), key = lambda x: x[1])
    delete_sub_leaf(tree, [new_leaf])
    return tree, val
    
def cutted_tree(tree_, X, target, mode_f, choose_f, verbose = 0):
    first_digits = lambda x: float(str(x)[:5])
    y = pd.to_numeric(X[target])
    tree = get_full_copy(tree_)
    best_metr = dict()
    best_tree = dict()
    y_pred = tree.predict(X, target = target)
    c = count_leaf(tree).shape[0]
    
    best_metr[c] = mode_f(y, y_pred)
    best_tree[c] = get_full_copy(tree)
    while (tree.edges.shape[0] > 0):
        tree, val = find_best_uncut(tree, X, y, target, mode_f, choose_f)
        c = count_leaf(tree).shape[0]
        best_metr[c] = val
        best_tree[c] = get_full_copy(tree)
    
    best_metric = first_digits(choose_f(best_metr.values()))
    min_leaf = min([k for k,v in best_metr.items() if first_digits(v) == best_metric])
    
    if verbose > 0:
        plt.clf()
        plt.plot(list(best_metr.keys()), list(best_metr.values()),'o')
        # plt.plot(list(best_metr.keys()), list(best_metr.values()),'b')
        plt.xlabel("Количество листов")#("Leafs")
        plt.ylabel(f"Лучшее значение метрики {mode_f.__name__}")# {target}")
        plt.title(f"Обрезка дерева по переменной {target}")
        plt.show()
        print(best_metr)
        print(best_metric, min_leaf)
    
    return best_tree[min_leaf]

class CRAID(object):
    """
    Survival decision tree model.

    Attributes
    ----------
    depth : int
        Maximal depth of nodes
    cut : boolean
        Flag of prunning
    features : list
        Available features
    categ : list
        Names of categorical features
    tree : Node 
        Root node of model
    coxph : CoxPHSurvivalAnalysis
        Model for hazard prediction
    ohenc : OneHotEncoder
        Encoding model from number of Node to indicators
    bins : array-like
        Points of timeline.
    info : dict
        Parameters for building nodes

    Methods
    -------
    
    fit : build decision tree with X, y data (recursive splitting node)
    predict : return values of features, rules or schemes
    predict_at_times : return survival or hazard function
    cut_tree : prunning function
    
    visualize : build graphviz Digraph for each node
    
    """
    def __init__(self, depth = 0,
                 random_state = 123,
                 features = [],
                 categ = [],
                 cut = False,
                 **info):
        self.info = info
        self.cut = cut
        self.remove_files = []
        self.tree = None
        self.depth = depth
        self.features = features
        self.categ = categ
        self.random_state = random_state
        self.name = "CRAID_%s" % (self.random_state)
        self.coxph = None
        self.ohenc = None
        self.bins = []
    
    def fit(self, X, y):
        if len(self.features) == 0:
            self.features = X.columns
        self.bins = cnt.get_bins(time = y[cnt.TIME_NAME])#, cens = y[cnt.CENS_NAME])
        X = X.reset_index(drop=True)
        X_tr = X.copy()
        X_tr[cnt.CENS_NAME] = y[cnt.CENS_NAME].astype(np.int32)
        X_tr[cnt.TIME_NAME] = y[cnt.TIME_NAME].astype(np.int32)
        
        if not("min_samples_leaf" in self.info):
            self.info["min_samples_leaf"] = 0.01*X_tr.shape[0]
        cnt.set_seed(self.random_state)
        
        if self.cut:
            X_val = X_tr.sample(n = int(0.2*X_tr.shape[0]), random_state=self.random_state)
            X_tr = X_tr.loc[X_tr.index.difference(X_val.index),:]
         
        # t_start = time.perf_counter()
        self.tree = Node(X_tr, features = self.features, 
                         categ = self.categ, **self.info)
        stack_nodes = np.array([self.tree], dtype = object)
        while(stack_nodes.shape[0] > 0):
            node = stack_nodes[0]
            stack_nodes = stack_nodes[1:]
            if node.depth >= self.depth:
                continue
            node.split()
            for ind_edge in range(node.edges.shape[0]):
                stack_nodes = np.append(stack_nodes, node.edges[ind_edge])
                
        if self.cut:
            self.cut_tree(X_val, cnt.CENS_NAME, mode_f = roc_auc_score, choose_f = max)
        
        self.coxph = CoxPHSurvivalAnalysis(alpha = 0.1)
        self.ohenc = OneHotEncoder(handle_unknown='ignore')
        pred_node = self.tree.predict(X, target = "num_node").to_numpy().reshape(-1,1)
        ohenc_node = self.ohenc.fit_transform(pred_node).toarray()
        self.coxph.fit(ohenc_node, y)
        
        # print(f"FULL_TIME: {time.perf_counter() - t_start} seconds")
        # print('*'*6, 'End fit.', '*'*6)
        return
    
    def predict(self, X, mode = "target", target = cnt.TIME_NAME, name_out = "res"):
        """
        Return values of features, rules or schemes

        Parameters
        ----------
        X : Pandas dataframe
            Contain input features of events.
        mode : str, optional
            Mode of predicting. The default is "target".
            "target" : return values of feature (in target variable)
            "scheme" : return all possible outcomes for features determination
                       target is list of features with missing values (relatively)
            "rules" : return full rules from node to leaf
        target : str or list, optional
            Aim of predicting. The default is occured time.
        name_out : str, optional
            Name of return column. The default is "res".

        Returns
        -------
        X[name_out] : array-like 
            Values by mode & target

        """
        X = format_to_pandas(X, self.features)
        X.loc[:, name_out] = np.nan
        if mode == "target":
            X.loc[:, name_out] = self.tree.predict(X, target)
        elif mode == "scheme":
            X.loc[:,'store_str'] = ""
            X.loc[:, name_out] = X[name_out].apply(lambda x: dict())
            X.loc[:, name_out] = self.tree.predict_scheme(X, target)
        elif mode == "rules":
            X.loc[:, name_out] = self.tree.predict_rules(X)
        return X[name_out]
    
    def predict_at_times(self, X, bins, mode = "surv"):
        """
        Return survival or hazard function.

        Parameters
        ----------
        X : Pandas dataframe
            Contain input features of events.
        bins : array-like
            Points of timeline.
        mode : str, optional
            Type of function. The default is "surv".
            "surv" : send building function in nodes
            "hazard" : fit CoxPH model on node numbers (input) 
                                          and time/cens (output)
                       predict cumulative HF from model 

        Returns
        -------
        array-like
            Vector of function values in times (bins).

        """
        X = format_to_pandas(X, self.features)
        if mode == "hazard":
            return self.predict_hazard(X, bins)
        def build_at_times(X_node):
            if mode == "surv":
                return metr.get_survival_func(X_node[cnt.TIME_NAME],
                                          X_node[cnt.CENS_NAME], 
                                          bins = bins)
            return metr.get_hazard_func(X_node[cnt.TIME_NAME],
                                          X_node[cnt.CENS_NAME], 
                                          bins = bins)
        X.loc[:,'f_at_times'] = np.nan
        X.loc[:,'f_at_times'] = self.tree.predict(X, build_at_times, name_tg = "f_at_times")
        return np.array(X['f_at_times'].to_list())
        
    def predict_hazard(self, X, bins):
        bins = np.clip(bins, self.bins.min(), self.bins.max())
        pred_node = self.tree.predict(X, target = "num_node").to_numpy().reshape(-1,1)
        ohenc_node = self.ohenc.transform(pred_node).toarray()
        hazards = self.coxph.predict_cumulative_hazard_function(ohenc_node)
        pred_haz = np.array(list(map(lambda x: x(bins), hazards)))
        return pred_haz
    
    def cut_tree(self, X, target, mode_f = roc_auc_score, choose_f = max):
        """
        Method of prunning tree.
        Find best subtree, which reaches best value of metric "mode_f""

        Parameters
        ----------
        X : Pandas dataframe
            Contain input features of events.
        target : str
            Feature name for metric counting.
        mode_f : function, optional
            Metric for selecting. The default is roc_auc_score.
        choose_f : function, optional
            Type of best value (max or min). The default is max.

        """
        self.tree = cutted_tree(self.tree, X, target, mode_f, choose_f)
    
    def visualize(self, path_dir = "", **kwargs):
        tmp_dir = path_dir + self.name + "\\"
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        dot = Digraph(node_attr={'shape': 'none'})
        dot = self.tree.build_viz(dot, path_dir = tmp_dir, **kwargs)
        dot.render(path_dir + self.name + "_", view = False, format = "png")
        for root, dirs, files in os.walk(tmp_dir):
            for file in files:
                os.remove(os.path.join(root, file))
        os.rmdir(tmp_dir)
        
    # def get_score(self, X_v, bins, mode = 'cox'):
    #     print("CRAID", end = " ")
    #     print("IBC: {:.4}".format(self.score(X_v, bins, name = "ibs")), end = " ")
    #     print("CONC: {:.4}".format(self.score(X_v, bins, name = "conc")))
        
    # def score(self, X_v, bins = None, name = "conc"):
    #     X_t = self.tree.get_df_node()
    #     if name == "conc":
    #         pred_time = self.tree.predict(X_v, cnt.TIME_NAME)
    #         return concordance_index(X_v[cnt.TIME_NAME],pred_time)
    #     pred_bins = self.predict_at_times(X_v, bins = bins, mode = "surv")
        
    #     y_train = cnt.get_y(X_t[cnt.CENS_NAME],X_t[cnt.TIME_NAME])
    #     y_true = cnt.get_y(X_v[cnt.CENS_NAME],X_v[cnt.TIME_NAME])
        
    #     return metr.ibs(y_train, y_true, pred_bins, bins)
    
    def translate(self, describe):
        self.features = [describe.get(f,f) for f in self.features]
        self.categ = [describe.get(c,c) for c in self.categ]
        self.tree.translate(describe)