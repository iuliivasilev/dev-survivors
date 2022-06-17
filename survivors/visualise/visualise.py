import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from .. import metrics as metr
from .. import criteria as scrit

def get_sch_top_descr(df, feats, thres = 5, top = 3):
    df['scheme'] = df.loc[:,feats].apply(lambda r: '*'.join(sorted([ni for ni in feats if r[ni] > 0])), axis = 1)
    
    d = pd.Series(df['scheme'].value_counts()).to_dict()
    #thres = min(thres, sorted(d.values())[-3:][0])
    d = {i:j for i,j in d.items() if j >= thres}
    top_3 = sorted(d, key = lambda x: d[x])[-top:][::-1]
    ret = {}
    for t in top_3:
        tmp = df[df['scheme'] == t]#.query(t + '==1')
        ret[t] = [tmp.shape[0],round(tmp['dDeath'].mean(),3),
                               round(tmp['cDeath'].mean(),3)#, TODO
                               # round(tmp['SUM_M'].mean(),3)
                               ]
    return ret

def get_sch_top_all_descr(df, feats, thres = 5, top = 3):
    df['scheme'] = df.loc[:,feats].apply(lambda r: '*'.join(sorted([ni for ni in feats if r[ni] > 0])), axis = 1)
    
    d = pd.Series(df['scheme'].value_counts()).to_dict()
    #thres = min(thres, sorted(d.values())[-3:][0])
    d = {i:j for i,j in d.items() if j >= thres}
    top_3 = sorted(d, key = lambda x: d[x])[-top:][::-1]
    ret = {}
    for t in top_3:
        tmp = df[df['scheme'] == t]#.query(t + '==1')
        ret[t] = [tmp.shape[0],tmp['dDeath'],
                                tmp['cDeath'] #, TODO
                               # tmp['SUM_M']
                               ]
    return ret

def prepare_name(k, descript):
    res = ""
    tmp = k.split('*')
    delete_shared = []
    for i, sch in enumerate(tmp):
        if '_'.join(tmp[:i] + tmp[i+1:]).find(sch) == -1:
            delete_shared.append(sch)
            
    res += '&'.join([descript.get(name, name) for name in delete_shared])
    return res

def get_full_descr_sch(dict_sch, descript):
    res = ""
    ind = 0
    for k,v in dict_sch.items():
        ind+=1
        # res += "СХЕМА " + str(ind) + ' (' + '  '.join(['КОЛ-ВО:'+str(v[0]),'ВЕРОЯТНОСТЬ СМЕРТИ:'+str(v[2]),'СРЕДНЯЯ СУММА:'+str(v[3])]) + ')\n'
        res += "СХЕМА " + str(ind) + ' (' + '  '.join(['КОЛ-ВО:'+str(v[0]),'ВЕРОЯТНОСТЬ СМЕРТИ:'+str(v[2])]) + ')\n'
        delete_shared = []
        tmp = k.split('*')
        for i, sch in enumerate(tmp):
            if '_'.join(tmp[:i] + tmp[i+1:]).find(sch) == -1:
                delete_shared.append(sch)
        res += '\n'.join([descript.get(name,name) for name in delete_shared])            
        res += '\n'
    return res

def export_legend(legend, filename):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

  
def visualize_pers_km(pred_dict, output_dir, descript, name = None):
    items = sorted(pred_dict,key=lambda x: np.mean(pred_dict[x][0]))[:5]
    plt.ioff()
    fig, ax = plt.subplots(figsize=(7, 5))
    legends = []
    i = 0
    for i in items:
        prob = np.mean(pred_dict[i][0])
        days = np.mean(pred_dict[i][1])
        # summ = pred_dict[i][2]
        prob_sch = pred_dict[i][2] #3 because SUM_M
        dict_sch = pred_dict[i][3] #4 because SUM_M
        kmf = metr.get_survival_func(pred_dict[i][1], pred_dict[i][0])
        ax = kmf.plot_survival_function(ax=ax, show_censors = True, censor_styles = {"marker":'o', "ms":6})
        if prob < 1.0:
            legends.append('Лечение завершено')
        descr = ""
        descr += 'ВЕРОЯТНОСТЬ СМЕРТИ:{:.5}'.format(str(prob)) + ' '
        # descr += 'СРЕДНЯЯ СУММА:{}'.format(str(summ))
        descr += '\n\n{}'.format(get_full_descr_sch(dict_sch, descript))#'\n'.join([get_full_sch(k) + ':' + '  '.join(['NUM:'+str(v[0]),'PROB:'+str(v[2]),'SUM:'+str(v[3])]) for k,v in dict_sch.items()]))
        legends.append(descr)
    plt.savefig(output_dir + 'full_figure.png')
    legend = ax.legend(legends, bbox_to_anchor = (0,-0.1), loc = 'upper left', ncol = 1, prop = {'size':10})
    export_legend(legend, output_dir + "full_legend.png")
    #plt.show()

def visualize_all_sh_km(pred_dict, output_dir, name = None):
    items = sorted(pred_dict,key=lambda x: np.mean(pred_dict[x][0]))[:5]
    plt.ioff()
    SCH_i = 1
    ind = 0
    legends = []
    fig, ax = plt.subplots(figsize=(7, 5))
    pict_dict = []
    for i in items:
        for sch in pred_dict[i][4].keys():
            ind += 1
            full_sch = pred_dict[i][4][sch]
            N = full_sch[0]
            dDeath = full_sch[1]
            cDeath = full_sch[2]
            # SUM_M = round(np.mean(full_sch[3]),3)
            prob = np.mean(cDeath)
            days = np.mean(dDeath)
            kmf = metr.get_survival_func(dDeath, cDeath)
            #descr = "Схема №" + str(ind)
            descr = "\nКОЛ-ВО:" + str(N) + ' '
            descr += 'ВЕРОЯТНОСТЬ СМЕРТИ:{:.5}'.format(str(prob)) + ' '
            # descr += 'СРЕДНЯЯ СУММА:{}'.format(str(SUM_M))
            descr += '\n' + prepare_name(sch)
            #    if prob < 1.0:
            #    legends.append('Лечение завершено')            
            if name is None:
                ax = kmf.plot_survival_function(ax=ax, censor_styles = {"marker":'o', "ms":6})#, show_censors = True)
                legends.append(descr)
            else:
                pict_dict.append([prob, N, kmf, descr])# pict_dict.append([prob, SUM_M, N, kmf, descr])
    if not(name is None):
        ind_sort = 0
        if name == 'sum':
            ind_sort = 1
        elif name == 'N':
            ind_sort = 2
        s_item = sorted(pict_dict, key = lambda x: x[ind_sort])
        if name == 'N':
            s_item = s_item[::-1]
        for s_i in s_item:
            ax = s_i[3].plot_survival_function(ax=ax, censor_styles = {"marker":'o', "ms":6})#, show_censors = True)
            legends.append(s_i[4])
    if ind > 0:
        add_name = "all_" if name is None else name + '_'
        plt.savefig(output_dir + add_name + "figure.png")
        legend = ax.legend(legends, bbox_to_anchor = (0,-0.1), loc = "upper left", ncol = 1, prop = {'size':10})
        export_legend(legend, output_dir + add_name + "legend.png")
        SCH_i += 1
    else:
        raise ValueError("Не выделены значимые схемы лечения")#"Не выделены значимые схемы лечения")
    #plt.show()
    
def visualize_descr_sh_km(pred_dict, 
                          output_dir, 
                          descript, name = None):
    items = sorted(pred_dict,key=lambda x: np.mean(pred_dict[x][0]))[:5]
    plt.ioff()
    ind = 0
    legends = []
    fig, ax = plt.subplots(figsize=(7, 5))
    pict_dict = []
    for i in items:
        for sch in pred_dict[i][4].keys():
            ind += 1
            full_sch = pred_dict[i][4][sch]
            N = full_sch[0]
            dDeath = full_sch[1]
            cDeath = full_sch[2]
            SUM_M = round(np.mean(full_sch[3]),3)
            prob = round(np.mean(cDeath),3)
            days = np.mean(dDeath)
            kmf = metr.get_survival_func(dDeath, cDeath)
            #descr = "Схема №" + str(ind)
            if name == "inter":
                ax = kmf.plot_survival_function(ax=ax, censor_styles = {"marker":'o', "ms":6})#, show_censors = True)
            elif name == "point": #ci_show 
                ax = kmf.plot_survival_function(ax=ax, censor_styles = {"marker":'o', "ms":6}, show_censors = True, ci_show = False)
                if prob < 1.0:
                    legends.append('Лечение завершено')
            elif name == "p&i":
                ax = kmf.plot_survival_function(ax=ax, censor_styles = {"marker":'o', "ms":6}, show_censors = True, ci_show = True)
                if prob < 1.0:
                    legends.append('Лечение завершено')
            elif name == "without":
                ax = kmf.plot_survival_function(ax=ax, censor_styles = {"marker":'o', "ms":6}, ci_show = False)
            name_plot = prepare_name(sch, descript)
            if name == "inter":
                pict_dict.append([ind, name_plot, prob, SUM_M, N])
            legends.append("СХЕМА №" + str(ind) + '\n' + name_plot.replace("&","\n"))
    if ind > 0:
        add_name = "all_" if name is None else name + '_'
        plt.savefig(output_dir + add_name + "figure.png")
        legend = ax.legend(legends, bbox_to_anchor = (0,-0.1), loc = "upper left", ncol = 1, prop = {'size':10})
        export_legend(legend, output_dir + "shared_legend.png")
        if name == "inter":
            descr = pd.DataFrame(pict_dict,columns = ['Номер','Описание','Вероятность','Стоимость','Количество'])
            descr.to_csv(output_dir + "SchDescr.csv", sep = ";",index = False)
    else:
        raise ValueError("Не выделены значимые схемы лечения")
    #plt.show()
    
def check_correct_list(pred_dict, join = False):
    base = dict()
    for i in pred_dict.keys():
        df = pd.DataFrame(pred_dict[i][2])
        # df = pd.DataFrame(pred_dict[i][3]) #TODO
        df['cDeath'] = pred_dict[i][0]
        df['dDeath'] = pred_dict[i][1]
        # df['SUM_M'] = pred_dict[i][2] #TODO
        base[i] = df
    if not join:
        return base
    while len(base.keys()) > 1:
        p_value_d = dict()
        for i1,l1 in enumerate(base.keys()):
            for i2,l2 in enumerate(base.keys()):
                if i2 > i1:
                    p_val = scrit.logrank_fast(base[l1]['dDeath'],base[l2]['dDeath'],
                                         base[l1]['cDeath'],base[l2]['cDeath'])
                    p_value_d[l1+'#'+l2] = p_val
        p_val_s = sorted(p_value_d, key = lambda x:p_value_d[x])
        print('Максимальное P-value:', p_value_d[p_val_s[-1]])
        if p_value_d[p_val_s[-1]] < 0.05:
            break
        f_l, s_l = p_val_s[-1].split('#')
        print('Цепочки схем:', f_l, s_l)
        print('Заменяются на:', f_l + '|' + s_l)
        base[f_l + '|' + s_l] = pd.concat([base[f_l].copy(),base[s_l].copy()])
        del base[f_l]
        del base[s_l]
    return base
        

# descript = cnt.mapp_sch
def pred_pers_km_count(pred_dict, output_dir, 
                       join = True, 
                       min_size = 20,
                       top_count = 100,
                       func_vis = visualize_pers_km, 
                       func_get_sch = get_sch_top_descr,
                       name = None,
                       descript = {}):
    res = dict()
    base = check_correct_list(pred_dict, join)
    for i in base.keys():
        df = base[i]
        scheme_feat = list(set(df.columns) - {'cDeath','dDeath'}) #TODO: ,'SUM_M'})
        tmp = [df['cDeath'],
               df['dDeath'],
               # round(np.mean(df['SUM_M']),2), #TODO
               {sch: df[sch].mean() for sch in scheme_feat},
               func_get_sch(df, scheme_feat, thres = min_size, top = top_count)
              ]
        if tmp[-1] != {}:
            res[i] = tmp
    func_vis(res, output_dir = output_dir, name = name, descript = descript)