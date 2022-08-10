import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Digraph
from joblib import Parallel, delayed

from .find_split import best_attr_split
from .. import constants as cnt
from .stratified_model import LeafModel
from ..scheme import Scheme

sns.set()

"""" Auxiliary functions """


def join_dict(a, b):
    return dict(list(a.items()) + list(b.items()))


class Rule(object):
    def __init__(self, feature: str, condition: str, has_nan: int):
        self.feature = feature
        self.condition = condition
        self.has_nan_ = has_nan

    def get_feature(self):
        return self.feature

    def get_condition(self):
        return self.condition

    def has_nan(self):
        return self.has_nan_

    def translate(self, describe):
        self.feature = describe.get(self.feature, self.feature)

    def to_str(self):
        s = f"({self.feature}{self.condition})"
        if self.has_nan_:
            s = f"({s}| nan)"  # не указано)"
        return s


""" Класс вершины дерева решений """
class Node(object):
    # __slots__ = ("df", "numb", "full_rule",
    #              "depth", "edges", "rule_edges", "features", "leaf_model",
    #              "categ", "woe", "is_leaf", "verbose", "info")

    def __init__(self, df,  numb=0, full_rule=[],
                 depth=0, features=[], categ=[], woe=False,
                 verbose=0, **info):
        self.df = df
        self.numb = numb
        self.full_rule = full_rule
        self.depth = depth
        self.edges = np.array([], dtype=object)
        self.rule_edges = np.array([], dtype=object)
        self.features = features
        self.categ = categ
        self.woe = woe
        self.is_leaf = True
        self.verbose = verbose
        self.info = info
        self.leaf_model = LeafModel()
        self.check_params()

    def check_params(self):
        self.info.setdefault("bonf", True)
        self.info.setdefault("n_jobs", 16)
        self.info.setdefault("max_features", 1.0)
        self.info.setdefault("signif", 1.1)
        self.info.setdefault("thres_cont_bin_max", 100)
        if self.info["max_features"] == "sqrt":
            self.info["max_features"] = int(np.trunc(np.sqrt(len(self.features))+0.5))
        elif isinstance(self.info["max_features"], float):
            self.info["max_features"] = int(self.info["max_features"]*len(self.features))
        self.leaf_model.fit(self.df)

    """ GROUP FUNCTIONS: CREATE LEAFS """

    def find_best_split(self):
        numb_feats = self.info["max_features"]
        numb_feats = np.clip(numb_feats, 1, len(self.features))
        n_jobs = min(numb_feats, self.info["n_jobs"])
        selected_feats = np.random.choice(self.features, size=numb_feats, replace=False)

        args = np.array([])
        for feat in selected_feats:
            t = self.info.copy()
            t["type_attr"] = ("woe" if self.woe else "categ") if feat in self.categ else "cont"
            t["arr"] = self.df.loc[:, [feat, cnt.CENS_NAME, cnt.TIME_NAME]].to_numpy().T
            args = np.append(args, t)
        with Parallel(n_jobs=n_jobs, verbose=0, batch_size=10) as parallel:
            ml = parallel(delayed(best_attr_split)(**a) for a in args)

        attrs = {f: ml[ind] for ind, f in enumerate(selected_feats)}
        attr = min(attrs, key=lambda x: attrs[x]["p_value"])

        if attrs[attr]["sign_split"] > 0 and self.info["bonf"]:
            attrs[attr]["p_value"] = attrs[attr]["p_value"] / attrs[attr]["sign_split"]
        return (attr, attrs[attr])

    def split(self):
        node_edges = np.array([], dtype=object)
        self.rule_edges = np.array([], dtype=object)

        attr, best_split = self.find_best_split()
        # The best split is not significant
        if best_split["sign_split"] == 0:
            if self.verbose > 0:
                print(f'Конец ветви, незначащее p-value: {best_split["p_value"]}')
            return node_edges

        if self.verbose > 0:
            print('='*6, best_split["p_value"], attr)
        for v, p_n in zip(best_split["values"], best_split["pos_nan"]):
            query = attr + v
            if p_n == 1:
                query = "(" + attr + v + ") or (" + attr + " != " + attr + ")"
            rule = Rule(feature=attr, condition=v, has_nan=p_n)
            d_node = self.df.query(query).copy()
            N = Node(df=d_node, full_rule=self.full_rule + [rule],
                     features=self.features, categ=self.categ,
                     depth=self.depth+1, verbose=self.verbose, **self.info)
            node_edges = np.append(node_edges, N)
            self.rule_edges = np.append(self.rule_edges, rule)

        return node_edges

    def set_edges(self, edges):
        self.edges = edges
        self.is_leaf = False
        self.df = None

    def set_leaf(self):
        if self.is_leaf:
            return
        self.edges = np.array([])
        self.is_leaf = True

    def prepare_df_for_attr(self, X):
        attr = self.rule_edges[0].get_feature()
        if attr not in X.columns:
            X.loc[:, attr] = np.nan
        return X[attr].to_numpy()

    def get_edges(self, X):
        X_np = self.prepare_df_for_attr(X)
        rule_id = 1 if self.rule_edges[0].has_nan() else 0
        query = self.rule_edges[rule_id].get_condition()
        if self.rule_edges[0].get_feature() in self.categ:
            values = np.isin(X_np, eval(query[query.find("["):]))
        else:
            values = eval("X_np" + query)
        return np.where(values, self.edges[rule_id], self.edges[1-rule_id])

    def get_full_rule(self):
        return " & ".join([s.to_str() for s in self.full_rule])

    def predict(self, X, target, bins=None):
        res = np.full((X.shape[0]), np.nan, dtype=object)
        if target == "surv":
            res = self.leaf_model.predict_survival_at_times(X, bins)  # target(X_node=dataset)
        elif target == "hazard":
            res = self.leaf_model.predict_hazard_at_times(X, bins)
        elif target in self.__dict__:
            res = np.repeat(getattr(self, target, np.nan), X.shape[0], axis=0)
        else:
            res = self.leaf_model.predict_mean_feature(X, target)  # np.mean(dataset[target])
        return res

    def predict_scheme(self, X, scheme_feats):
        feat_means = np.array([self.leaf_model.features_mean.get(s_f, np.nan)
                               for s_f in scheme_feats])
        times = self.leaf_model.survival.durations
        cens = self.leaf_model.survival.event_observed

        return Scheme(self.get_full_rule(), times, cens, feat_means)

    """ GROUP FUNCTIONS: VISUALIZATION """

    def get_figure(self, mode="hist", bins=None, target=cnt.CENS_NAME, save_path=""):
        plt.ioff()
        fig, ax = plt.subplots(figsize=(8, 6))
        if mode == "hist":
            lst = self.leaf_model.predict_list_feature(target)
            plt.hist(lst, bins=25)
            ax.set_xlim([0, np.max(lst)])
            ax.set_xlabel(f'{target}', fontsize=25)
        elif mode == "surv":
            sf = self.leaf_model.predict_survival_at_times(X=None, bins=bins)
            plt.step(bins, sf)
            ax.set_xlabel('Time', fontsize=25)
            ax.set_ylabel('Survival probability', fontsize=25)
        plt.savefig(save_path)
        plt.close(fig)

    def get_description(self):
        m_cens = round(self.leaf_model.predict_mean_feature(X=None, feature_name=cnt.CENS_NAME), 2)
        m_time = round(self.leaf_model.predict_mean_feature(X=None, feature_name=cnt.TIME_NAME), 2)
        label = "\n".join([f"size = {self.leaf_model.get_shape()[0]}",
                           f"cens/size = {m_cens}",
                           f"depth = {self.depth}",
                           f"death = {m_time}"])
        return label

    def set_dot_node(self, dot, path_dir="", depth=None, **args):
        if not(depth is None) and depth < self.depth :
            return dot
        img_path = path_dir + str(self.numb) + '.png'
        self.get_figure(save_path=img_path, **args)
        dot.node(str(self.numb), label=self.get_description(),
                 image=img_path, fontsize='30')  # fontsize='16'
        return dot

    def set_dot_edges(self, dot):
        if not(self.is_leaf):
            for e in range(len(self.rule_edges)):
                s = self.rule_edges[e].to_str()
                dot.edge(str(self.numb), str(self.edges[e]), label=s, fontsize='30')
        return dot

    def translate(self, describe):
        if self.is_leaf:
            self.df = self.df.rename(describe, axis=1)
        self.features = [describe.get(f, f) for f in self.features]
        self.categ = [describe.get(c, c) for c in self.categ]
        for e in range(len(self.rule_edges)):
            self.rule_edges[e].translate(describe)
