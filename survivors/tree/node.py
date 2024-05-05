import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed
# from scipy.stats import chi2
from ..criteria import chi2_isf

from .find_split_hist import hist_best_attr_split
from .. import constants as cnt
from ..external import LEAF_MODEL_DICT, LeafModel
from ..scheme import Scheme

custom_params = {"axes.spines.right": False, 'grid.color': 'lightgray', 'axes.grid': True, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
# custom_params = {"font.size": 25, "axes.labelsize": 25, "xtick.labelsize": 25, "ytick.labelsize": 25,
#                  "axes.spines.right": False, 'grid.color': 'lightgray', 'axes.grid': True, "axes.spines.top": False}

""" Auxiliary functions """


class Rule(object):
    """
    Node of decision tree.
    Allow to separate data into 2 child nodes

    Attributes
    ----------
    feature : str
        Name of feature for splitting
    condition : str
        Operation for splitting
    has_nan_ : bool
        Flag of the missing values in node

    Methods
    -------
    get_feature : Return feature
    get_condition : Return condition
    translate: Replace rule by dictionary
    to_str : Transforming to linear form
    print : Print all attributes and descriptions

    """
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

    def translate(self, describe: dict):
        """
        Rename feature in rule
        """
        self.feature = describe.get(self.feature, self.feature)

    def to_str(self):
        s = f"({self.feature}{self.condition})"
        if self.has_nan_:
            s = f"({s}| nan)"
        return s

    def print(self):
        """
        Print all attributes and descriptions
        """
        print(f"has_nan: {self.has_nan()}")
        print(f"feature: {self.get_feature()}")
        print(f"condition: {self.get_condition()}")


class Node(object):
    """
    Node of decision tree.
    Allow to separate data into 2 child nodes

    Attributes
    ----------
    df : Pandas DataFrame
        Data of the Node
    numb : int
        Number or name of the Node
    full_rule : list
        Rules from the root
    depth : int
        Distance from the root
    edges : array-like
        Numbers of child nodes
    rule_edges : array-like
        Rules for child nodes
    features : list
        Available features
    categ : list
        Names of categorical features
    woe : boolean
        Mode of categorical preparation
    is_leaf : boolean
        True if node is terminal (there are no child nodes)
    verbose : int
        Print the best split of the node
    info : dict
        Parameters for finding the best splitting
    leaf_model : LeafModel
        Stratified model by parameter <<leaf_model>> (map with LEAF_MODEL_DICT)

    Methods
    -------
    check_params : Fill empty parameters and map max_features to int
    find_best_split : Choose best split of node according to parameters
    split : Find best split of df sample and create child nodes
    set_edges: Set number of child nodes from hash table of main tree
    set_leaf : Delete child nodes and set node as terminal

    predict : Return statistic values of a data
    predict_scheme : Return all possible outcomes for additional features determination

    prepare_df_for_attr : set input values to numpy format (and fill missing features)
    get_edges : defines appropriate child nodes according to input values
    get_full_rule : convert full_rules to a string format

    get_figure : Create picture of a data (hist, survival function)
    get_description : Return common values of a data (size, depth, death, cens)

    set_dot_node : add self node to the graphviz dot
    set_dot_edges : add child nodes to the graphviz dot
    translate : Replace rules and features by dictionary

    """
    def __init__(self, df,  numb=0, full_rule=[],
                 depth=0, features=[], categ=[], woe=False,
                 verbose=0, **info):
        self.df = df
        self.numb = numb
        self.size = None
        self.full_rule = full_rule
        self.depth = depth
        self.edges = np.array([], dtype=int)
        self.rule_edges = np.array([], dtype=Rule)
        self.features = features
        self.categ = categ
        self.woe = woe
        self.is_leaf = True
        self.verbose = verbose
        self.info = info
        self.check_params()

    def check_params(self):
        self.info.setdefault("bonf", True)
        self.info.setdefault("max_features", 1.0)
        self.info.setdefault("signif", 1.1)
        self.info.setdefault("signif_stat", chi2_isf(min(self.info["signif"], 1.0), df=1))
        self.info.setdefault("thres_cont_bin_max", 100)
        self.info.setdefault("normalize", True)

        if self.info["max_features"] == "sqrt":
            self.info["max_features"] = int(np.trunc(np.sqrt(len(self.features))+0.5))
        elif isinstance(self.info["max_features"], float):
            self.info["max_features"] = int(self.info["max_features"] * len(self.features))

        self.info.setdefault("weights_feature", None)
        if not (self.info["weights_feature"] is None):
            self.info["weights"] = self.df[self.info["weights_feature"]].to_numpy()

        leaf_kwargs = {k[5:]: v for k, v in self.info.items() if (k.find("leaf_") != -1) and (k != "leaf_model")}
        self.info.setdefault("leaf_model", "base_zero_after")  # base
        if isinstance(self.info["leaf_model"], str):
            self.leaf_model = LEAF_MODEL_DICT.get(self.info["leaf_model"], "base_zero_after")(**leaf_kwargs)  # base
        elif isinstance(self.info["leaf_model"], type):  # Check is class
            self.leaf_model = self.info["leaf_model"](**leaf_kwargs)
            if not (isinstance(self.leaf_model, LeafModel)):
                self.leaf_model = None
        else:
            self.leaf_model = None
        self.size = self.df.shape[0]

        self.leaf_model.fit(self.df)
        # self.ch = np.array(
        #     [np.mean(self.df["time"]), np.std(self.df["time"]), np.sum(self.df["cens"]) / self.df["cens"].shape[0]])

    """ GROUP FUNCTIONS: CREATE LEAVES """
    def get_comb_fast(self, features):
        """
        Create set of all triplets with two target variables and one splitting feature
        """
        d_df = dict(zip(self.df.columns, self.df.values.T))

        def create_params_f(name):
            d = self.info.copy()
            d["arr"] = np.vstack((d_df[name], d_df[cnt.CENS_NAME], d_df[cnt.TIME_NAME]))
            d["type_attr"] = ("woe" if self.woe else "categ") if name in self.categ else "cont"
            return d

        return list(map(create_params_f, features))

    def find_best_split(self):
        """
        Sort through all combinations of splitting the sample by features.
        Find a split with the highest statistical value.
        """
        numb_feats = self.info["max_features"]
        numb_feats = np.clip(numb_feats, 1, len(self.features))
        n_jobs = self.info.get("n_jobs", 1 if numb_feats < 20 else 5)

        selected_feats = list(np.random.choice(self.features, size=numb_feats, replace=False))

        args = self.get_comb_fast(selected_feats)
        # ml = np.vectorize(lambda x: hist_best_attr_split(**x))(args)
        with Parallel(n_jobs=n_jobs, verbose=self.verbose) as parallel:
            ml = parallel(delayed(hist_best_attr_split)(**a) for a in args)

        attrs = {f: ml[ind] for ind, f in enumerate(selected_feats)}

        # attr = min(attrs, key=lambda x: attrs[x]["p_value"])  # best by p-value
        attr = max(attrs, key=lambda x: attrs[x]["stat_val"])  # best by stat-val
        # attrs_gr = dict(filter(lambda x: x[1]["sign_split"] > 0, attrs.items()))
        # if len(attrs_gr) == 0:
        #     attr = min(attrs, key=lambda x: attrs[x]["p_value"])
        # else:
        #     attr = min(attrs_gr, key=lambda x: attrs_gr[x]["p_value"])
        #     if self.info["bonf"]:
        #         attrs[attr]["p_value"] = attrs[attr]["p_value"] / attrs[attr]["sign_split"]

        # if attrs[attr]["sign_split"] > 0 and self.info["bonf"]:  # suffix for simple p-value
        #     attrs[attr]["p_value"] = attrs[attr]["p_value"] / attrs[attr]["sign_split"]
        return (attr, attrs[attr])

    def ind_for_nodes(self, X_attr, best_split, is_categ):
        """
        Map the number of the according child node by rule and sample features.
        """
        rule_id = best_split["pos_nan"].index(0)
        query = best_split["values"][rule_id]
        if is_categ:
            values = np.isin(X_attr, eval(query[query.find("["):]))
        else:
            values = eval("X_attr" + query)
        return np.where(values, rule_id, 1 - rule_id)

    def split(self):
        """
        Find best split of df sample and create child nodes
        """
        node_edges = np.array([], dtype=int)
        self.rule_edges = np.array([], dtype=Rule)

        attr, best_split = self.find_best_split()
        # The best split is not significant
        if best_split["sign_split"] == 0:
            if self.verbose > 0:
                print(f'The node is terminal, best p-value: {best_split["stat_val"]}')
            return node_edges
        if self.verbose > 0:
            print('='*self.depth, best_split["stat_val"], attr)

        branch_ind = self.ind_for_nodes(self.df[attr], best_split, attr in self.categ)

        for n_b in np.unique(branch_ind):
            rule = Rule(feature=attr,
                        condition=best_split["values"][n_b],
                        has_nan=best_split["pos_nan"][n_b])
            d_node = self.df[branch_ind == n_b].copy()
            N = Node(df=d_node, full_rule=self.full_rule + [rule],
                     features=self.features, categ=self.categ,
                     depth=self.depth + 1, verbose=self.verbose, **self.info)
            node_edges = np.append(node_edges, N)
            self.rule_edges = np.append(self.rule_edges, rule)

        if self.rule_edges.shape[0] == 1:
            print(branch_ind, self.df[attr], best_split, attr in self.categ)
            raise ValueError('ERROR: Only one branch created!')

        self.df = None
        return node_edges

    def set_edges(self, edges):
        self.edges = edges
        self.is_leaf = False

    def set_leaf(self):
        self.edges = np.array([], dtype=int)
        self.is_leaf = True
        self.df = None

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
        """
        Predict target values for X data

        Parameters
        ----------
        X : Pandas dataframe
            Contain input features of events
        target : str or function
            Column name, mode or aggregate function of a leaf sample
            Column name : must be in dataset.columns
            - Return mean of feature
            Mode :
            - "surv" return survival function
            - "hazard" return cumulative hazard function
            - attribute_name return attribute value (e.g. depth, numb)
            - feature_name return aggregate statistic value for node
        bins : array-like
            Points of timeline

        Returns
        -------
        res : array-like
            Values by target
        """
        if target == "surv":
            return self.leaf_model.predict_survival_at_times(X, bins)
        elif target == "hazard":
            return self.leaf_model.predict_hazard_at_times(X, bins)
        # elif target == "ch":
        #     return np.repeat(self.ch[np.newaxis, :], X.shape[0], axis=0)
        elif target in self.__dict__:
            return np.repeat(getattr(self, target, np.nan), X.shape[0], axis=0)
        return self.leaf_model.predict_feature(X, target)

    def predict_scheme(self, X, scheme_feats):
        feat_means = np.array([self.leaf_model.features_predict.get(s_f, np.nan)
                               for s_f in scheme_feats])
        times = self.leaf_model.predict_list_feature(cnt.TIME_NAME)
        cens = self.leaf_model.predict_list_feature(cnt.CENS_NAME)

        return Scheme(self.get_full_rule(), times, cens, feat_means)

    """ GROUP FUNCTIONS: VISUALIZATION """

    def get_figure(self, mode="hist", bins=None, target=cnt.CENS_NAME, save_path=""):
        plt.ioff()
        fig, ax = plt.subplots(figsize=(7, 5))
        if mode == "hist":
            lst = self.leaf_model.predict_list_feature(target)
            plt.hist(lst, bins=25)
            ax.set_xlim([0, np.max(lst)])
            ax.set_xlabel(f'{target}', fontsize=25)
        elif mode == "kde":
            lst = self.leaf_model.predict_list_feature(target)
            # lst = self.leaf_model.old_durs
            sns.kdeplot(lst, ax=ax)
            ax.set_xlabel(f'{target}', fontsize=25)
        elif mode == "surv":
            sf = self.leaf_model.predict_survival_at_times(X=None, bins=bins)
            if len(sf.shape) > 1:
                sf = sf.flat
            plt.step(bins, sf, linewidth=3)
            ax.set_xlabel('Time', fontsize=25)
            ax.set_ylabel('Survival probability', fontsize=25)
        plt.savefig(save_path)
        plt.close(fig)

    def get_description(self):
        m_cens = self.leaf_model.predict_feature(X=None, feature_name=cnt.CENS_NAME)
        m_time = self.leaf_model.predict_feature(X=None, feature_name=cnt.TIME_NAME)
        if isinstance(m_cens, np.ndarray):
            m_cens = m_cens[0]
        if isinstance(m_time, np.ndarray):
            m_time = m_time[0]
        label = "\n".join([f"size = {self.leaf_model.get_shape()[0]}",
                           f"events (%) = {round(m_cens, 2)}",
                           # f"depth = {self.depth}",
                           f"mean time = {round(m_time, 2)}"])
        return label

    def set_dot_node(self, dot, path_dir="", depth=None, **args):
        if not (depth is None) and depth < self.depth:
            return dot
        img_path = path_dir + str(self.numb) + '.png'
        self.get_figure(save_path=img_path, **args)
        dot.node(str(self.numb), label=self.get_description(),
                 image=img_path, fontsize='30')  # fontsize='16'
        return dot

    def set_dot_edges(self, dot):
        if not self.is_leaf:
            for e in range(len(self.rule_edges)):
                s = self.rule_edges[e].to_str()
                dot.edge(str(self.numb), str(self.edges[e]), label=s, fontsize='30')
        return dot

    def translate(self, describe):
        self.features = [describe.get(f, f) for f in self.features]
        self.categ = [describe.get(c, c) for c in self.categ]
        for e in range(len(self.rule_edges)):
            self.rule_edges[e].translate(describe)
