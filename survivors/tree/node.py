import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Digraph
from joblib import Parallel, delayed

from .find_split import best_attr_split
from .. import metrics as metr
from .. import constants as cnt

sns.set()

""" Auxiliary functions """
def join_dict(a, b):
    return dict(list(a.items()) + list(b.items()))


class LeafModel(object):
    def __init__(self):
        self.survival = None
        self.hazard = None
        self.features_mean = dict()

    def fit(self, X_node):
        self.survival = metr.get_survival_func(X_node[cnt.TIME_NAME], X_node[cnt.CENS_NAME])
        self.hazard = metr.get_hazard_func(X_node[cnt.TIME_NAME], X_node[cnt.CENS_NAME])
        self.features_mean = X_node.mean(axis=0).to_dict()

    def predict_mean_feature(self, X, feature_name):
        return self.features_mean[feature_name]

    def predict_survival_at_times(self, X, bins):
        return self.survival.survival_function_at_times(bins).to_numpy()

    def predict_hazard_at_times(self, X, bins):
        return self.survival.cumulative_hazard_at_times(bins).to_numpy()


""" Класс вершины дерева решений """
class Node(object):
    """
    Node of decision tree.
    Allow to separate data into 2 subnodes (references store in edges) 

    Attributes
    ----------
    df : Pandas DataFrame
        Data of Node
    numb : int
        Number or name of Node
    depth : int
        Distance from root node
    edges : array-like
        Subbranches after separating
    features : list
        Available features
    categ : list
        Names of categorical features
    woe : boolean
        Mode of categorical preparation
    rule : dict
        Allow to define data to node.
        name: str
            query in pandas terms
        attr: str
            feature of separation
        pos_nan: int
            Indicator of nan
    is_leaf : boolean
        True if node don't have subnodes
    verbose : int
        Print best split of node
    info : dict
        Parameters for finding the best split

    Methods
    -------
    check_params : Fill empty parameters and map max_features to int
    find_best_split : Choose best split of node according to parameters
    split : Try to create subnodes by best split
    get_df_node : Return data for node
    set_leaf : Delete subnodes and reset data
    
    predict : Return statistic values of data
    predict_rules : Return full rules from node to leaf
    predict_scheme : Return all possible outcomes for additional features determination
    
    get_figure : Create picture of data (hist, survival function)
    get_rule : Return rule of node
    get_description : Return common values of data (size, depth, death, cens)
    build_viz : Create and fill graphviz digraph
    translate : Replace rules and features by dictionary
    
    """
    __slots__ = ("df", "numb",
                 "depth", "edges", "features", "leaf_model",
                 "categ", "woe", "rule", "is_leaf", "verbose", "info")

    def __init__(self, df,  numb=1, depth=0,
                 features=[], categ=[], woe=False,
                 rule={"name": "", "attr": "", "pos_nan": 0},
                 verbose=0, **info):
        self.df = df
        self.numb = numb
        self.depth = depth
        self.edges = np.array([])
        self.features = features
        self.categ = categ
        self.woe = woe
        self.rule = rule
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
            t["type_attr"] = "woe" if self.woe else "categ" if feat in self.categ else "cont"
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
        attr, best_split = self.find_best_split()
        # В лучшем признаке не было ни одного значимого разбиения
        if best_split["sign_split"] == 0:
            if self.verbose > 0:
                print(f'Конец ветви, незначащее p-value: {best_split["p_value"]}')
            return (attr, best_split)
        
        if self.verbose > 0:
            print('='*6, best_split["p_value"], attr)
        leaf_ind = 0
        for v, p_n in zip(best_split["values"], best_split["pos_nan"]):
            query = attr + v
            if p_n == 1:
                query = "(" + attr + v + ") or (" + attr + " != " + attr + ")"
            d_node = self.df.query(query).copy()
            N = Node(df=d_node,
                     features=self.features, categ=self.categ,
                     depth=self.depth+1, numb=self.numb*2+leaf_ind,
                     rule={"name": attr + v, "attr": attr, "pos_nan": p_n},
                     verbose=self.verbose, **self.info)
            self.edges = np.append(self.edges, N)
            leaf_ind += 1
        self.is_leaf = False
        self.df = None
        
    """ GROUP FUNCTIONS: CLEAR AND DEL """
    def get_df_node(self):
        if self.is_leaf:
            return self.df
        return pd.concat([edge.get_df_node() for edge in self.edges])
    
    def set_leaf(self):
        if self.is_leaf:
            return
        self.df = self.get_df_node()
        del self.edges
        self.edges = np.array([])
        self.is_leaf = True
        
    """ GROUP FUNCTIONS: PREDICT """
        
    def predict(self, X, target, name_tg="res", bins=None, end_list=[]):
        """
        Return statistic values of data

        Parameters
        ----------
        X : Pandas dataframe
            Contain input features of events.
        target : str or function
            Column name, mode or aggregate function of leaf sample.
            Column name : must be in dataset.columns
                Return mean of feature
            Mode :
                "surv" return survival function
                "hazard" return cumulative hazard function
                "depth" return leafs depth
                "num_node" return leafs numb (names)
        bins : array-like
            Points of timeline
        name_tg : str, optional
            Name of return column. The default is "res".
        end_list : list, optional
            Numbers of end node (instead leaf). The default is [].

        Returns
        -------
        X[name_tg] : array-like 
            Values by target

        """
        if (self.numb in end_list) or self.is_leaf:
            if target == "surv" or target == "hazard":
                if target == "surv":
                    func_at_times = self.leaf_model.predict_survival_at_times(X, bins)  # target(X_node=dataset)
                else:
                    func_at_times = self.leaf_model.predict_hazard_at_times(X, bins)
                X.loc[:, name_tg] = X[name_tg].apply(lambda x: func_at_times)
            elif target == "depth":
                X.loc[:, name_tg] = self.depth
            elif target == "num_node":
                X.loc[:, name_tg] = self.numb
            else:
                dataset = self.get_df_node()
                if target in dataset.columns:
                    X.loc[:, name_tg] = self.leaf_model.predict_mean_feature(X, target)  # np.mean(dataset[target])
        else:
            attr = self.edges[0].rule['attr']
            if attr not in X.columns:
                X.loc[:, attr] = np.nan
            ind_nan = X.index
            for edge in self.edges:
                ind_nan = ind_nan.difference(X.query(edge.rule["name"]).index)
                
            for edge in self.edges:
                ind = X.query(edge.rule["name"]).index
                if edge.rule["pos_nan"] == 1:
                    ind = ind.append(ind_nan)
                if len(ind) > 0:
                    X.loc[ind, name_tg] = edge.predict(X=X.loc[ind, :], target=target, bins=bins,
                                                       name_tg=name_tg, end_list=end_list)
        return X[name_tg]
    
    def predict_rules(self, X, name_tg="res"):
        if self.is_leaf:
            X.loc[:, name_tg] = self.get_rule()
        else:
            attr = self.edges[0].rule['attr']
            if attr not in X.columns:
                X.loc[:, attr] = np.nan
            ind_nan = X.query(attr + "!=" + attr).index
            for edge in self.edges:
                ind = X.query(edge.rule["name"]).index
                if edge.rule["pos_nan"] == 1:
                    ind = ind.append(ind_nan)
                if len(ind) > 0:
                    X.loc[ind, name_tg] = edge.predict_rules(X.loc[ind, :], name_tg)
                    if len(self.rule["name"]) > 0:
                        X.loc[ind, name_tg] = self.get_rule() + '&' + X.loc[ind, name_tg]
        return X[name_tg]
    
    def get_values_column(self, columns):
        return [0, 1]
    
    def predict_scheme(self, X, scheme_feat):
        """
        Return all possible outcomes for additional features determination

        Parameters
        ----------
        X : Pandas dataframe
            Contain input features of events.
        scheme_feat : list
            Features with missing values (relatively).
            If feature in list was used in node, 
                then node consider all possible replaces for value in branches
                Thus, method allow to return all outcomes for different values

        Returns
        -------
        X['res'] : array-like
            For each observation contain dict
                key : rule of overdefined feature (; is separator)
                value : list of sample values
                    censoring flag, time, all values for scheme_feat
                    
        """
        def scheme_output_format(r):
            to_array = lambda col: np.array(self.df.get(col))
            return {r['store_str']:
                    [to_array(cnt.CENS_NAME), 
                     to_array(cnt.TIME_NAME), 
                     # to_array(self.info["sum"]), # TODO FOR SCHEME'S SUM
                     {sch: to_array(sch) for sch in scheme_feat}]}
            
        def join_scheme_leafs(X_sub, ind_nan=[]):
            for edge in self.edges:
                ind = X_sub.query(edge.rule['name']).index
                if len(ind_nan) > 0:
                    if edge.rule["pos_nan"] == 1:
                        ind = ind.append(ind_nan)
                if len(ind) > 0:
                    X_sub.loc[ind, 'tmp'] = edge.predict_scheme(X_sub.loc[ind, :], scheme_feat)
                    X_sub.loc[ind, 'res'] = X_sub.loc[ind, :].apply(lambda r: join_dict(r['res'], r['tmp']), axis=1)
            return X_sub['res']
            
        if self.is_leaf:
            return X.apply(scheme_output_format, axis=1)
        attr = self.edges[0].rule['attr']
        if attr not in X.columns:
            X.loc[:, attr] = np.nan
        ind_nan = X.query(attr + "!=" + attr).index
        ind_has = X.index.difference(ind_nan)
        if attr not in scheme_feat:
            X.loc[:, 'res'] = join_scheme_leafs(X, ind_nan)
        else:
            if len(ind_has) > 0:
                X.loc[ind_has, 'res'] = join_scheme_leafs(X.loc[ind_has, :])
            if len(ind_nan) > 0:
                pred_store = X.loc[ind_nan, 'store_str'].copy()
                for val in self.get_values_column(attr):
                    X.loc[ind_nan, attr] = val
                    X.loc[ind_nan, 'store_str'] = pred_store + attr + '==' + str(val) + ';'
                    X.loc[ind_nan, 'res'] = join_scheme_leafs(X.loc[ind_nan, :])
        return X['res']
    
    """ GROUP FUNCTIONS: VISUALIZATION """
    
    def get_figure(self, mode="hist", target=None, save_path=""):
        if len(save_path) > 0:
            plt.ioff()
        fig, ax = plt.subplots(figsize=(8, 6))
        local_df = self.get_df_node()
        if mode == "hist":
            local_df[target].hist(bins=25)
            ax.set_xlim([0, np.max(local_df[target])])
        elif mode == "surv":
            kmf = metr.get_survival_func(local_df[cnt.TIME_NAME], local_df[cnt.CENS_NAME])
            ax.set_xlim([0, np.max(local_df[cnt.TIME_NAME])])
            ax.set_ylim([0, 1])
            plt.xticks(range(0, np.max(local_df[cnt.TIME_NAME])+1, 1000))
            kmf.plot_survival_function(legend=False, fontsize=25)
            # ax.set_xlabel('Время', fontsize=25)
            # ax.set_ylabel('Вероятность выживания', fontsize=25)
            ax.set_xlabel('Time', fontsize=25)
            ax.set_ylabel('Survival probability', fontsize=25)  # plt.xlabel('Timeline', fontsize=0)
        if len(save_path) > 0:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def get_rule(self):
        if not(self.rule["pos_nan"]):
            return f'({self.rule["name"]})'
        # return f'(({self.rule["name"]})|({self.rule["attr"]} != {self.rule["attr"]}))'
        return f'(({self.rule["name"]})| не указано)'
    
    def get_description(self, full=False):
        s = ""  # if not(self.rule["pos_nan"]) else " or " + self.rule["attr"] + " == NaN"
        d = self.get_df_node()
        m_cens = round(d[cnt.CENS_NAME].mean(), 2)
        m_time = round(d[cnt.TIME_NAME].mean(), 2)
        if full:
            label = "\n".join([self.rule["name"] + s,
                               "size = %s" % (d.shape[0]),
                               "cens/size = %s" % (m_cens),
                               "depth = %s" % (self.depth),
                               "death = %s" % (m_time)])
        else:
            label = self.rule["name"] + s 
        return label
        
    def build_viz(self, dot=None, path_dir="", depth=None, **args):
        if dot is None:
            dot = Digraph()
        img_path = path_dir + str(self.numb) + '.png'
        self.get_figure(save_path=img_path, **args)
        dot.node(str(self.numb), label=self.get_description(),
                 image=img_path, fontsize='30')  # fontsize='16'
        if not(depth is None):
            if depth < self.depth:
                return dot
        for ind_e in range(self.edges.shape[0]):
            dot = self.edges[ind_e].build_viz(dot, path_dir, **args)
            dot.edge(str(self.numb), str(self.edges[ind_e].numb))
        return dot
         
    def translate(self, describe):
        if self.is_leaf:
            self.df = self.df.rename(describe, axis=1)
        self.features = [describe.get(f, f) for f in self.features]
        self.categ = [describe.get(c, c) for c in self.categ]
        self.rule["name"] = describe.get(self.rule["name"], self.rule["name"])
        for edge in self.edges:
            edge.translate(describe)
