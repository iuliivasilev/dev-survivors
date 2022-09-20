import numpy as np
import random
from .. import metrics as metr
from .. import criteria as scrit
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class Scheme(object):
    def __init__(self, rule, times, cens, feat_means):
        self.rule = rule
        self.times = times
        self.cens = cens
        self.feat_means = feat_means
        self.shape = times.shape[0]
        self.kmf = metr.get_survival_func(self.times, self.cens)

    def join(self, sch):
        self.rule = f"({self.rule})|({sch.rule})"
        self.times = np.hstack([self.times, sch.times])
        self.cens = np.hstack([self.cens, sch.cens])
        self.feat_means = (self.feat_means + sch.feat_means) / 2
        return self

    def predict_proba(self):
        return round(np.mean(self.cens), 5)

    def predict_time(self):
        return round(np.mean(self.times), 5)

    def copy(self):
        return Scheme(self.rule, self.times, self.cens, self.feat_means)

    def get_str_rules(self):
        return self.rule

    def get_description(self):
        descr = "\nКОЛ-ВО:{self.shape} "
        descr += f"ВЕРОЯТНОСТЬ СОБЫТИЯ:{self.predict_proba()} "
        descr += f"ВРЕМЯ СОБЫТИЯ:{self.predict_time()} "
        descr += "\n" + self.get_str_rules()
        return descr

    def get_subschemes(self, min_size=5, top=3):
        ret = [self.copy()]
        return ret

    def plot_sf(self, ax, show_censors=False, ci_show=False):
        return self.kmf.plot_survival_function(ax=ax, censor_styles={"marker": 'o', "ms": 6},
                                               show_censors=show_censors, ci_show=ci_show)


class FilledSchemeStrategy(object):
    def __init__(self, schemes_list):
        self.schemes_dict = {sch.get_str_rules(): sch for sch in schemes_list}

    def join(self, fss):
        for k, v in fss.schemes_dict.items():
            if k in self.schemes_dict.keys():
                self.schemes_dict[k].join(v)
            else:
                self.schemes_dict[k] = v

    def join_nearest_leaves(self, sign_thres=0.05, diff_func=random.random):
        def delete_k_from_dict(d, del_k):
            d_ = dict()
            for k_ in d.keys():
                if not (del_k in k_.split("#")):
                    d_[k_] = d[k_]
            return d_

        base = self.schemes_dict
        diff_dict = dict()
        for i1, l1 in enumerate(base.keys()):
            for i2, l2 in enumerate(base.keys()):
                if i2 > i1:
                    diff_dict[l1 + '#' + l2] = scrit.logrank(base[l1].times, base[l2].times, base[l1].cens,
                                                             base[l2].cens)
        while len(base) > 1:
            max_pair_key, min_stat_val = min(diff_dict.items(), key=lambda x: x[1])
            max_p_val = stats.chi2.sf(min_stat_val, df=1)
            print('Максимальное P-value:', max_p_val)
            if max_p_val < sign_thres:
                break
            f_l, s_l = max_pair_key.split('#')
            new_sch = base[f_l].copy().join(base[s_l])
            new_sch_name = new_sch.get_str_rules()
            for k in [f_l, s_l]:
                diff_dict = delete_k_from_dict(diff_dict, k)
                del base[k]
            for k in base.keys():
                diff_dict[new_sch_name + '#' + k] = scrit.logrank(new_sch.times, base[k].times, new_sch.cens,
                                                                  base[k].cens)
            base[new_sch_name] = new_sch
            print('Цепочки схем:', f_l, s_l)
            print('Заменяются на:', new_sch_name)
        self.schemes_dict = base

    def get_flatten_sch(self, sort_by=None):
        schemes = []
        for k, v in self.schemes_dict.items():
            schemes += v.get_subschemes()
        # descending order
        if sort_by == "size":
            schemes.sort(key=lambda x: x.shape, reverse=True)
        elif sort_by == "proba":
            schemes.sort(key=lambda x: x.predict_proba(), reverse=True)
        elif sort_by == "time":
            schemes.sort(key=lambda x: x.predict_time(), reverse=True)
        return schemes

    def predict_best_scheme(self, sort_by="proba"):
        return self.get_flatten_sch(sort_by=sort_by)[0]

    def visualize_schemes(self, output_dir=None, sort_by=None, show_censors=False, ci_show=False):
        fig, ax = plt.subplots(figsize=(7, 5))
        schemes = self.get_flatten_sch(sort_by=sort_by)
        ind = 0
        legends = []

        for sch in schemes:
            ind += 1
            descr = f"Схема №{ind}" + sch.get_description()
            ax = sch.plot_sf(ax, show_censors=show_censors, ci_show=ci_show)
            if show_censors and (0 in sch.cens):
                legends.append("Лечение завершено")
            legends.append(descr)
            if ci_show:
                legends.append("Доверительный интервал")

        if len(legends) > 0:
            ax.get_legend().remove()
            if not (output_dir is None):
                plt.savefig(output_dir + "scheme_figure.png")
            legend = ax.legend(legends, bbox_to_anchor=(0, -0.1), loc="upper left", ncol=1, prop={'size': 10})
            fig_leg = legend.figure
            fig_leg.canvas.draw()
            bbox = legend.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
            if not (output_dir is None):
                fig_leg.savefig(output_dir + "scheme_legend.png", dpi="figure", bbox_inches=bbox)
                plt.close()
            else:
                plt.show()
        else:
            raise ValueError("Не выделены значимые схемы лечения")
