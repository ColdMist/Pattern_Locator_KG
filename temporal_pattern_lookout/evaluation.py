# -*- coding: utf-8 -*- 
# @Time : 2023/1/27 15:00 
# @Author : Yinan 
# @File : evaluation.py
import os

import pandas as pd
from utilities import PlotTools
from utilities import AnalysisTools
import matplotlib.pyplot as plt
import matplotlib.ticker as ticke

# TODO: 1. r1 --> # symmetric num of r1 --> # percentage of symmetric of r1
# TODO: 2. for certain relation, count conclusion in test set and premise in train set.


def plot_distribution(on='symmetric'):
    stat_path = '../results/icews14/statistics'
    temp_path = stat_path + '/dynamic/{}/'.format(on)
    static_path = stat_path + '/static/{}/'.format(on)

    ploter = PlotTools()
    for file_ord in [temp_path, static_path]:
        data_path = list()
        for file in os.listdir(file_ord):
            if file[-3:] == 'csv':
                data_path.append(file)

        test, train = pd.read_csv(file_ord + data_path[0], index_col=0), pd.read_csv(file_ord + data_path[1], index_col=0)
        ploter.plot_distribution_rel(train, test, showall=True
                                     , dynamic=True if file_ord == temp_path else False
                                     , on=on
                                     , save_path=file_ord)


def pattern_analyse(dataset, pattern):
    for by in ['train', 'test']:
        analyser = AnalysisTools()
        save_path = '../results/{}/statistics/{}'.format(dataset, by)
        path_list = {'s_%s' % pattern: save_path + '/static'
            , 's_rel': save_path + '/static'
            , 'd_%s' % pattern: save_path + '/dynamic'
            , 'd_rel': save_path + '/dynamic'
                     }

        for p in path_list.values():
            if not os.path.exists(p):
                os.makedirs(p)

        if pattern.split()[0] == 'temporal':
            set_t_pattern = pd.read_csv(r'../results/{}/pattern sets/{}/set {}.csv'
                                        .format(dataset, by, pattern))
            stat_t_rel = pd.read_csv(r'../results/{}/pattern sets/{}/stat_t_rel.csv'
                                     .format(dataset, by))
            stat_t_pattern = analyser.occurance_pattern(set_t_pattern, stat_t_rel, '%s' % pattern)

            # save states fo symmetric
            stat_t_pattern.reset_index(drop=True).to_csv(
                '{}/freq_{}.csv'.format(path_list['d_%s' % pattern], pattern),
                index=False)

            stat_t_rel.reset_index(drop=True).rename(columns={'number': 'freq'}).to_csv(
                '{}/freq_rel.csv'.format(path_list['d_rel']), index=False)
        else:
            set_pattern = pd.read_csv(r'../results/{}/pattern sets/{}/set {}.csv'
                                        .format(dataset, by, pattern))
            stat_rel = pd.read_csv(r'../results/{}/pattern sets/{}/stat_rel.csv'
                                   .format(dataset, by))
            stat_pattern = analyser.occurance_pattern(set_pattern, stat_rel, '%s' % pattern)

            # save states fo symmetric
            stat_pattern.reset_index(drop=True).to_csv('{}/freq_{}.csv'.format(path_list['s_%s' % pattern], pattern),
                                                   index=False)
            stat_rel.reset_index(drop=True).rename(columns={'number': 'freq'}).to_csv(
                                '{}/freq_rel.csv'.format(path_list['s_rel']), index=False)


def conclusion_premise_paar(train_set, test_set, dataset, pattern):
    save_path = '../results/{}/statistics/con_pre_pair'.format(dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    test = pd.read_csv(test_set)
    test_reversed = test.copy()
    if pattern == 'symmetric' or pattern == 'temporal symmetric':
        test_reversed.loc[:, 'head'], test_reversed.loc[:, 'tail'] = \
            test_reversed.loc[:, 'tail'].copy(), test_reversed.loc[:, 'head'].copy()

        train = pd.read_csv(train_set)
        intersected = pd.merge(train, test_reversed, how='inner')
        int_vc = intersected.value_counts('relation')

        test_vc = test.value_counts('relation')
        stat = pd.DataFrame((int_vc / test_vc))

        stat.insert(0, 'number in test set', test_vc)
        stat.insert(1, 'number of symmetric in train set', int_vc)
        stat.fillna(0, inplace=True)
        stat.rename(columns={0: 'percentage'}, inplace=True)
        stat['number of symmetric in train set'] = stat['number of symmetric in train set'].astype(int)
        stat.sort_values(by='number in test set', inplace=True, ascending=False)
        stat.to_csv(save_path + '/{}.csv'.format(pattern))

    # if pattern == 'inverse' or pattern == 'temporal inverse':

def plot_distribution(dataset, pattern, by='freq'):
    for t in ['train', 'test']:
        typ = 'dynamic' if pattern.split()[0] == 'temporal' else 'static'
        data = pd.read_csv('../results/{}/statistics/{}/{}/freq_{}.csv'.format(dataset, t, typ, pattern))
        fig, ax = plt.subplots(2, 1, figsize=(24, 6), sharex=True)
        if by == 'freq':
            ax[0].bar(data.loc[:, 'relation'].apply(str), data.loc[:, 'number of relation'],
                   label='number of relation', color='#f9766e', edgecolor='grey', alpha=0.5)
            ax[0].bar(data.loc[:, 'relation'].apply(str), data.loc[:, 'number of %s' % pattern],
                   label='number of %s' % pattern, color='#00bfc4', edgecolor='grey', alpha=0.5)
            ax[1].bar(data.loc[:, 'relation'].apply(str), data.loc[:, 'percentage'],
                   label='percentage', color='#7B68EE', edgecolor='grey', alpha=0.7)

            ax[0].set_xlabel('relations', fontsize=12)
            ax[1].set_xlabel('relations', fontsize=12)
            ax[0].set_ylabel('number', fontsize=12)
            ax[1].set_ylabel('percentage', fontsize=12)
            for i in range(2):
                ax[i].tick_params(axis='x', length=0, rotation=30)
                ax[i].grid(axis='y', alpha=0.5, ls='--')
                ax[i].legend(frameon=False)
                # ax.set_title('{} {}'.format('Dynamic' if dynamic else 'Static', on))

            fig.suptitle('({}) {}-{}-{}'.format(dataset, t, typ, pattern))
            plt.xticks([])
            # fig.legend(data.loc[:5, 'relation'])
            save_path = '../results/{}/statistics/{}/{}/'.format(dataset, t, typ)
            plt.savefig(save_path + '%s_Distribution.png' % pattern, dpi=300)
            plt.show()


def main(dataset, pattern):
    # plot
    # for on in ['symmetric', 'relations']:
    #     plot_distribution(on=on)

    pattern_analyse(dataset, pattern)
    # inverse_analyse(dataset)
    conclusion_premise_paar('../results/{}/pattern sets/train/set {}.csv'.format(dataset,  pattern),
                        '../results/{}/pattern sets/test/set {}.csv'.format(dataset, pattern),  dataset, pattern)

    plot_distribution(dataset, pattern)

