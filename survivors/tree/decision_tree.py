import pandas as pd
import numpy as np
import os
import copy
import tempfile

from sklearn.metrics import roc_auc_score

from .node import Node
from ..scheme import FilledSchemeStrategy
from .. import constants as cnt
import matplotlib.pyplot as plt

from graphviz import Digraph, set_jupyter_format
set_jupyter_format('png')


def format_to_pandas(X, columns):
    type_df = type(X)
    if type_df.__name__ == "DataFrame":
        return X[columns].reset_index(drop=True)
    elif type_df.__name__ == "ndarray":
        return pd.DataFrame(X, columns=columns)
    return None


def get_oversample(df, target=cnt.CENS_NAME):
    from imblearn.over_sampling import RandomOverSampler

    oversample = RandomOverSampler(sampling_strategy='minority', random_state=42)
    df_over, _ = oversample.fit_resample(df, df[target])
    return df_over


""" Functions of prunning """


def find_best_uncut(tree, X, y, target, mode_f, choose_f):
    span_leaf = tree.get_spanning_leaf_numbers()
    d = {}
    for el in span_leaf:
        y_pred = tree.predict(X, target=target, end_list=[el])
        d[el] = round(mode_f(y, y_pred), 4)

    new_leaf, val = choose_f(d.items(), key=lambda x: x[1])
    tree.delete_leaves_by_span([new_leaf])
    return tree, val


def cutted_tree(tree_, X, target, mode_f, choose_f, verbose=0):
    first_digits = lambda x: float(str(x)[:5])
    y = pd.to_numeric(X[target])
    tree = copy.deepcopy(tree_)
    best_metr = dict()
    best_tree = dict()
    y_pred = tree.predict(X, target=target)
    c = tree.get_leaf_numbers().shape[0]

    best_metr[c] = mode_f(y, y_pred)
    best_tree[c] = copy.deepcopy(tree)
    while len(tree.nodes) > 1:
        tree, val = find_best_uncut(tree, X, y, target, mode_f, choose_f)
        c = tree.get_leaf_numbers().shape[0]
        best_metr[c] = val
        best_tree[c] = copy.deepcopy(tree)

    best_metric = first_digits(choose_f(best_metr.values()))
    min_leaf = min([k for k, v in best_metr.items() if first_digits(v) == best_metric])

    if verbose > 0:
        plt.clf()
        plt.plot(list(best_metr.keys()), list(best_metr.values()), 'o')
        # plt.plot(list(best_metr.keys()), list(best_metr.values()), 'b')
        plt.xlabel("Количество листов")  # ("Leafs")
        plt.ylabel(f"Лучшее значение метрики {mode_f.__name__}")  # {target}")
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
    nodes : dict
        Dictionary of all tree's nodes (numbers from hierarchy)
    cut : boolean
        Flag of pruning
    balance : boolean
        Flag of source data balancing
    depth : int
        Maximal depth of nodes
    features : list
        Available features
    categ : list
        Names of categorical features
    random_state : int
        Fixed seed for building reproducibility
    name : str
        Model's name
    bins : array-like
        Points of timeline.
    info : dict
        Parameters for building nodes

    Methods
    -------

    fit : build decision tree with X, y data (iterative splitting node)
    predict : return values of features, rules or schemes
    predict_at_times : return survival or hazard function
    predict_schemes : return FilledSchemeStrategy or Scheme
    cut_tree : pruning function

    visualize : build graphviz Digraph for each node
    translate : Replace rules and features by dictionary

    get_leaf_numbers : return leaf numbers from nodes
    get_spanning_leaf_numbers : return pre-leaves numbers from nodes
    delete_leaves_by_span : set up pre-leaves from lists to leaves

    """
    def __init__(self, depth=0,
                 random_state=123,
                 features=[],
                 categ=[],
                 cut=False,
                 balance=None,
                 **info):
        self.info = info
        self.cut = cut
        self.balance = balance
        self.nodes = dict()
        self.depth = depth
        self.features = features
        self.categ = categ
        self.random_state = random_state
        self.name = f"CRAID_{self.random_state}"
        self.bins = []

    def update_params(self, X_tr):
        if not ("min_samples_leaf" in self.info):
            self.info["min_samples_leaf"] = 0.01
        if isinstance(self.info["min_samples_leaf"], float):
            self.info["min_samples_leaf"] = max(int(self.info["min_samples_leaf"] * X_tr.shape[0]), 1)

        cnt.set_seed(self.random_state)

        if self.balance in ["balance", "balance+correct"]:
            freq = X_tr[cnt.CENS_NAME].value_counts()
            self.correct_proba = freq[1] / (freq[1] + freq[0])  # or freq[1] / (freq[0])

            X_tr = get_oversample(X_tr, target=cnt.CENS_NAME)
        elif self.balance in ["balance+weights"]:
            freq = X_tr[cnt.CENS_NAME].value_counts()

            X_tr["weights_obs"] = np.where(X_tr[cnt.CENS_NAME], freq[0] / freq[1], 1)
            self.info["weights_feature"] = "weights_obs"
        elif self.balance in ["too_late"]:
            X_tr[cnt.CENS_NAME][X_tr[cnt.TIME_NAME] > np.quantile(X_tr[cnt.TIME_NAME], 0.7)] = False

        elif self.balance in ["only_log_rank"]:
            self.info["balance"] = True

        if "l_reg" in self.info:
            self.info["apr_time"] = X_tr[cnt.TIME_NAME]
            self.info["apr_event"] = X_tr[cnt.CENS_NAME]

    def fit(self, X, y):
        if len(self.features) == 0:
            self.features = X.columns
        self.bins = cnt.get_bins(time=y[cnt.TIME_NAME])  # , cens = y[cnt.CENS_NAME])
        X = X.reset_index(drop=True)
        X_tr = X.copy()
        X_tr[cnt.CENS_NAME] = y[cnt.CENS_NAME].astype(np.int32)
        X_tr[cnt.TIME_NAME] = y[cnt.TIME_NAME].astype(np.float32)

        self.update_params(X_tr)

        if self.cut:
            X_val = X_tr.sample(n=int(0.2 * X_tr.shape[0]), random_state=self.random_state)
            X_tr = X_tr.loc[X_tr.index.difference(X_val.index), :]

        self.nodes[0] = Node(X_tr, features=self.features, categ=self.categ, **self.info)
        stack_nodes = np.array([0], dtype=int)
        while stack_nodes.shape[0] > 0:
            node = self.nodes[stack_nodes[0]]
            stack_nodes = stack_nodes[1:]
            if node.depth >= self.depth:
                node.set_leaf()
                continue
            sub_nodes = node.split()
            if sub_nodes.shape[0] > 0:
                sub_numbers = np.array([len(self.nodes) + i for i in range(sub_nodes.shape[0])])
                for i in range(sub_nodes.shape[0]):
                    sub_nodes[i].numb = sub_numbers[i]
                self.nodes.update(dict(zip(sub_numbers, sub_nodes)))
                node.set_edges(sub_numbers)
                stack_nodes = np.append(stack_nodes, sub_numbers)

        if self.cut:
            self.cut_tree(X_val, cnt.CENS_NAME, mode_f=roc_auc_score, choose_f=max)

        return

    def predict(self, X, mode="target", target=cnt.TIME_NAME, end_list=[], bins=None):
        """
        Return values by mode & target

        Parameters
        ----------
        X : Pandas dataframe
            Contain input features of events.
        mode : str, optional
            Mode of predicting. The default is "target".
            "surv" : return values of survival function in bins
            "hazard" : return values of hazard function in bins
            "target" : return values of feature (in target variable)
            "rules" : return full rules from node to leaf
        target : str or list, optional
            An aim of predicting. The default is occurred time.
        end_list : list, optional
            Numbers of endpoint nodes (for cutting)
        bins : array-like, optional
            Points of timeline

        Returns
        -------
        res : array-like
            Values by mode & target

        """
        X = format_to_pandas(X, self.features)
        num_node_to_key = dict(zip(sorted(self.nodes.keys()), range(len(self.nodes))))
        node_bin = np.zeros((X.shape[0], len(self.nodes)), dtype=bool)
        node_bin[:, 0] = 1
        shape = (X.shape[0])
        if not (bins is None):
            shape = (X.shape[0], len(bins))
        if target == "ch":
            shape = (X.shape[0], 3)
        if mode == "rules":
            res = np.full(shape, np.nan, dtype=object)
        else:
            # res = np.full(shape, np.nan, dtype=float)
            res = np.empty(shape, dtype=float)
            res[:] = np.nan
        for i in sorted(self.nodes.keys()):
            ind = node_bin[:, num_node_to_key[i]]
            if np.any(ind):
                X_loc = X[ind]
                if self.nodes[i].is_leaf or (i in end_list):
                    if target == "surv" or target == "hazard":
                        res[ind] = self.nodes[i].predict(X_loc, target, bins)
                    elif mode == "target":
                        res[ind] = self.nodes[i].predict(X_loc, target)
                    elif mode == "rules":
                        res[ind] = self.nodes[i].get_full_rule()
                else:
                    pred_edges = self.nodes[i].get_edges(X_loc)
                    for e in set(pred_edges):
                        node_bin[ind, num_node_to_key[e]] = pred_edges == e
        return res

    def predict_at_times(self, X, bins, mode="surv"):
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
            "hazard" : send building function in nodes

        Returns
        -------
        res : array-like
            Vector of function values in times (bins)

        """
        X = format_to_pandas(X, self.features)
        pred = self.predict(X, target=mode, bins=bins)  # Reverse correction
        if self.balance == "balance+correct":
            if mode == "hazard":
                pred = pred * self.correct_proba
            elif mode == "surv":
                pred = pred ** self.correct_proba
        return pred

    def predict_schemes(self, X, scheme_feats):
        X = format_to_pandas(X, self.features)
        num_node_to_key = dict(zip(sorted(self.nodes.keys()), range(len(self.nodes))))
        node_bin = np.zeros((X.shape[0], len(self.nodes)), dtype=bool)
        node_bin[:, 0] = 1
        for i in sorted(self.nodes.keys()):
            i_num = num_node_to_key[i]
            ind = np.where(node_bin[:, i_num])[0]
            ind_x = X.index[ind]
            if ind.shape[0] > 0:
                if not self.nodes[i].is_leaf:
                    if self.nodes[i].rule_edges[0].get_feature() in scheme_feats:
                        for e in self.nodes[i].edges:
                            node_bin[ind, num_node_to_key[e]] = 1
                    else:
                        pred_edges = self.nodes[i].get_edges(X.loc[ind_x, :])
                        for e in set(pred_edges):
                            node_bin[ind, num_node_to_key[e]] = pred_edges == e
        leaf_keys = self.get_leaf_numbers()
        leaf_numb = np.array([num_node_to_key[leaf] for leaf in leaf_keys])

        ret_leaf_numbers = np.where(node_bin[:, leaf_numb], leaf_keys, np.inf)

        dict_leaf_scheme = {n_l: self.nodes[n_l].predict_scheme(None, scheme_feats)
                            for n_l in np.unique(ret_leaf_numbers) if n_l != np.inf}
        dict_str_fill = {}
        for leaf_list in np.unique(ret_leaf_numbers, axis=0):
            end_leaf = leaf_list[leaf_list != np.inf]
            sch_list = np.vectorize(dict_leaf_scheme.get)(end_leaf)
            dict_str_fill[str(end_leaf)] = FilledSchemeStrategy(sch_list)

        res = np.array(list(map(lambda leaf_list: dict_str_fill.get(str(leaf_list[leaf_list != np.inf]), np.nan),
                                ret_leaf_numbers)), dtype=object)
        return res

    def cut_tree(self, X, target, mode_f=roc_auc_score, choose_f=max):
        """
        Method of pruning tree.
        Find the best subtree that achieves the best value of the "mode_f" metric".

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
        self.nodes = cutted_tree(self, X, target, mode_f, choose_f).nodes

    def visualize(self, path_dir=None, **kwargs):
        """
        Build graphviz representation of the tree

        Parameters
        ----------
        path_dir : str
        kwargs : dict
            Keyword arguments for nodes

        Returns
        -------
        dot : graphviz.Digraph
        """
        if path_dir is None:
            path_dir = os.getcwd()
        kwargs["bins"] = self.bins

        with tempfile.TemporaryDirectory() as tmp_dir:
            dot = Digraph(node_attr={'shape': 'none'})
            ordered_nodes = sorted(self.nodes.keys())
            for i in ordered_nodes:
                dot = self.nodes[i].set_dot_node(dot, path_dir=tmp_dir, **kwargs)
            for i in ordered_nodes:
                dot = self.nodes[i].set_dot_edges(dot)
            dot.render(os.path.join(path_dir, self.name), cleanup=True, format="png")
        return dot

    def translate(self, describe):
        """ Rename features for each node by dictionary """
        self.features = [describe.get(f, f) for f in self.features]
        self.categ = [describe.get(c, c) for c in self.categ]
        for i in self.nodes.keys():
            self.nodes[i].translate(describe)

    """ Cutting methods """

    def get_leaf_numbers(self):
        return np.array([i for i in self.nodes.keys() if self.nodes[i].is_leaf])

    def get_spanning_leaf_numbers(self):
        """ Get pre-termination nodes (have two leaves in edges) """
        leaves = self.get_leaf_numbers()
        return np.array([i for i in self.nodes.keys()
                         if np.intersect1d(self.nodes[i].edges, leaves).shape[0] == 2])

    def delete_leaves_by_span(self, list_span_leaf):
        """ Set pre-termination nodes as leaves """
        for i in list_span_leaf:
            for e in self.nodes[i].edges:
                del self.nodes[e]
            self.nodes[i].set_leaf()

    def tolerance_find_best(self, *args, **kwargs):
        pass
