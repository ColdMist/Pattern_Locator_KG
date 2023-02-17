# -*- coding: utf-8 -*- 
# @Time : 2023/1/24 19:24 
# @Author : Yinan 
# @File : utilities.py
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticke
import numpy as np
from itertools import combinations, permutations

# a) in training set: freq of symmetric grounding per relation
# b) total freq of samples for each relations
# c) compute percentage
# d) distubution of training and teat set
# e) put a threshold to select symmetric etc
# f) for static symmetric: count how many samples by that relation exist in test
# another: for dynamic symmetric: select top k dynamic symmetric relations based on trainig you already got,
# then count how many of test samples their Premise exist in train  ,premise->conclusion


class AnalysisTools:
    def __init__(self, dataset, pattern):
        self.dataset = dataset
        self.pattern = pattern

        if self.pattern == 'evolve' or self.pattern.split()[0] == 'temporal':
            self.temporal = True
        else:
            self.temporal = False

        self.save_path = '../results/{}/statistics/'.format(self.dataset)
        self.static_rel = pd.read_csv('../results/{}/pattern sets/train/stat_rel.csv'.format(self.dataset))
        self.temporal_rel = pd.read_csv('../results/{}/pattern sets/train/stat_t_rel.csv'.format(self.dataset))

    @ staticmethod
    def occurance_pattern(target_pattern, stat_rel, pattern_name):
        counts_num = stat_rel.copy().set_index('relation')
        counts_num.sort_index(inplace=True)
        pattern_set = pd.DataFrame(target_pattern.value_counts('relation')).rename(columns={0: 'number'})
        stat_rel = stat_rel.set_index('relation', inplace=False)
        stat = (pattern_set / stat_rel).fillna(0).rename(columns={'number': 'percentage'})
        stat.insert(0, 'number of relation', counts_num.loc[:, 'number'])
        stat.insert(1, 'number of {}'.format(pattern_name), pattern_set.loc[:, 'number'])
        stat = stat.fillna(0)
        stat.reset_index(inplace=True)
        stat['number of {}'.format(pattern_name)] = stat['number of {}'.format(pattern_name)].astype(int)
        return stat.sort_values(by='number of relation', ascending=False)

    def pattern_analyse(self):
        for by in ['train', 'test']:
            save_path = self.save_path + by
            path_list = {'s_%s' % self.pattern: save_path + '/static'
                , 's_rel': save_path + '/static'
                , 'd_%s' % self.pattern: save_path + '/dynamic'
                , 'd_rel': save_path + '/dynamic'
                              }
            for p in path_list.values():
                if not os.path.exists(p):
                    os.makedirs(p)

            if self.temporal:
                set_t_pattern = pd.read_csv(r'../results/{}/pattern sets/{}/set {}.csv'
                                            .format(self.dataset, by, self.pattern))
                stat_t_rel = pd.read_csv(r'../results/{}/pattern sets/{}/stat_t_rel.csv'
                                         .format(self.dataset, by))
                stat_t_pattern = self.occurance_pattern(set_t_pattern, stat_t_rel, '%s' % self.pattern)

                # save states fo symmetric
                stat_t_pattern.reset_index(drop=True).to_csv(
                    '{}/freq_{}.csv'.format(path_list['d_%s' % self.pattern], self.pattern),
                    index=False)

                stat_t_rel.reset_index(drop=True).rename(columns={'number': 'freq'}).to_csv(
                    '{}/freq_rel.csv'.format(path_list['d_rel']), index=False)
            else:
                set_pattern = pd.read_csv(r'../results/{}/pattern sets/{}/set {}.csv'
                                            .format(self.dataset, by, self.pattern))
                stat_rel = pd.read_csv(r'../results/{}/pattern sets/{}/stat_rel.csv'
                                       .format(self.dataset, by))
                stat_pattern = self.occurance_pattern(set_pattern, stat_rel, '%s' % self.pattern)

                # save states fo symmetric
                stat_pattern.reset_index(drop=True).to_csv('{}/freq_{}.csv'.format(path_list['s_%s' % self.pattern], self.pattern),
                                                       index=False)
                stat_rel.reset_index(drop=True).rename(columns={'number': 'freq'}).to_csv(
                                    '{}/freq_rel.csv'.format(path_list['s_rel']), index=False)

    def pattern_pair_analyse(self):
        def cal_comb(relations: pd.Series) -> list:
            return list(combinations([i for i in relations], 2))

        def pattern_pair(pattern_set: pd.DataFrame) -> pd.DataFrame:
            relations_stat = self.temporal_rel if self.temporal else self.static_rel
            dic = pd.DataFrame(columns=['%s i->j' % self.pattern, '#%s' % self.pattern, '#rel_i', 'percentage'])
            check = set()
            rel_list = cal_comb(pattern_set.loc[:, 'relation'].drop_duplicates().reset_index(drop=True))

            for rel in rel_list:
                if rel not in check and rel[-1::-1] not in check:
                    check.add(rel)
                    check.add(rel[-1::1])
                    rel1 = pattern_set.copy()[pattern_set['relation'] == rel[0]]
                    rel2 = pattern_set.copy()[pattern_set['relation'] == rel[1]]
                    if self.pattern in ['inverse', 'temporal inverse']:
                        rel2.loc[:, 'head'], rel2.loc[:, 'tail'] = rel2.loc[:, 'tail'].copy(), rel2.loc[:, 'head'].copy()
                    if self.pattern == 'evolve':
                        # TODO： how to count evolve?
                        ans = pd.merge(rel1, rel2, how='inner', on=['head', 'tail'])
                        p_rel1 = ans[ans['time_x'] <= ans['time_y']].shape[0]
                        p_rel2 = ans[ans['time_x'] >= ans['time_y']].shape[0]
                    else:
                        ans = pd.merge(rel1, rel2, how='inner',
                                   on=['head', 'tail', 'time'] if self.temporal else ['head', 'tail'])
                        p_rel1 = p_rel2 = ans.shape[0]
                    num_rel1 = int(relations_stat[relations_stat['relation'] == rel[0]]['number'].values)
                    num_rel2 = int(relations_stat[relations_stat['relation'] == rel[1]]['number'].values)
                    temp1 = pd.DataFrame([[rel, p_rel1, num_rel1, p_rel1 / num_rel1]]
                                         , columns=['%s i->j' % self.pattern, '#%s' % self.pattern, '#rel_i', 'percentage'])
                    temp2 = pd.DataFrame([[rel[-1::-1], p_rel2, num_rel2, p_rel2 / num_rel2]]
                                         , columns=['%s i->j' % self.pattern, '#%s' % self.pattern, '#rel_i', 'percentage'])
                    dic = pd.concat([dic, temp1, temp2], axis=0)
            return dic

        save_path = self.save_path + 'train'
        path_list = {'s_%s' % self.pattern: save_path + '/static'
            , 's_rel': save_path + '/static'
            , 'd_%s' % self.pattern: save_path + '/dynamic'
            , 'd_rel': save_path + '/dynamic'
                              }
        for p in path_list.values():
            if not os.path.exists(p):
                os.makedirs(p)

        if self.temporal:
            set_t_pattern = pd.read_csv(r'../results/{}/pattern sets/train/set {}.csv'
                                        .format(self.dataset, self.pattern))

            res = pattern_pair(set_t_pattern)
            res.to_csv('{}/pair_{}.csv'.format(path_list['d_%s' % self.pattern], self.pattern),
                    index=False)

        else:
            set_pattern = pd.read_csv(r'../results/{}/pattern sets/train/set {}.csv'
                                      .format(self.dataset, self.pattern))
            res = pattern_pair(set_pattern)
            res.to_csv('{}/pair_{}.csv'.format(path_list['s_%s' % self.pattern], self.pattern),
                           index=False)

    def conclusion_premise_paar(self):
        save_path = '../results/{}/statistics/con_pre_pair'.format(self.dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        test_set = '../results/{}/pattern sets/test/set {}.csv'.format(self.dataset, self.pattern)
        train_set = '../results/{}/pattern sets/train/set {}.csv'.format(self.dataset, self.pattern)
        test = pd.read_csv(test_set)
        test_reversed = test.copy()
        if self.pattern == 'symmetric' or self.pattern == 'temporal symmetric':
            test_reversed.loc[:, 'head'], test_reversed.loc[:, 'tail'] = \
                test_reversed.loc[:, 'tail'].copy(), test_reversed.loc[:, 'head'].copy()

            train = pd.read_csv(train_set)
            intersected = pd.merge(train, test_reversed, how='inner')
            int_vc = intersected.value_counts('relation')

            test_vc = test.value_counts('relation')
            stat = pd.DataFrame((int_vc / test_vc))

            stat.insert(0, 'number in test set', test_vc)
            stat.insert(1, 'number of {} in train set'.format(self.pattern), int_vc)
            stat.fillna(0, inplace=True)
            stat.rename(columns={0: 'percentage'}, inplace=True)
            stat['number of {} in train set'.format(self.pattern)] = stat[
                'number of {} in train set'.format(self.pattern)].astype(int)
            stat.sort_values(by='number in test set', inplace=True, ascending=False)
            stat.to_csv(save_path + '/{}.csv'.format(self.pattern))




class PlotTools:
    def __init__(self, dataset):
        self.dataset = dataset

    @staticmethod
    def plot_distribution_rel(train_set, test_set, showall=False, dynamic=False, on='symmetric', save_path=None):
        if not showall:
            train_set = train_set[~train_set['freq'].isin([0])]
            test_set = test_set[~test_set['freq'].isin([0])]
        train_set.reset_index(inplace=True)
        test_set.reset_index(inplace=True)
        fig, ax = plt.subplots(1, 1, figsize=(24, 6))
        if on == 'relations':
            ax.bar(train_set.loc[:, 'relation'].apply(str), train_set.loc[:, 'freq'] / train_set.loc[:, 'freq'].sum(),
                   label='Train', color='#f9766e', edgecolor='grey', alpha=0.5)
            ax.bar(test_set.loc[:, 'relation'].apply(str), test_set.loc[:, 'freq'] / test_set.loc[:, 'freq'].sum(),
                   label='Test', color='#00bfc4', edgecolor='grey', alpha=0.5)
        elif on == 'symmetric':
            ax.bar(train_set.loc[:, 'relation'].apply(str), train_set.loc[:, 'freq'],
                   label='Train', color='#f9766e', edgecolor='grey', alpha=0.5)
            ax.bar(test_set.loc[:, 'relation'].apply(str), test_set.loc[:, 'freq'],
                   label='Test', color='#00bfc4', edgecolor='grey', alpha=0.5)
        ax.set_xlabel('relations', fontsize=12)
        ax.set_ylabel('percentage', fontsize=12)
        ax.tick_params(axis='x', length=0, rotation=30)
        ax.grid(axis='y', alpha=0.5, ls='--')
        ax.legend(frameon=False)
        ax.set_title('{} {}'.format('Dynamic' if dynamic else 'Static', on))

        if on == 'relations':
            ax.xaxis.set_major_locator(ticke.MultipleLocator(base=5))

        if save_path:
            plt.savefig(save_path + 'Distribution.png', dpi=300)
        # plt.show()

    def plot_train_test_distribution(self):
        train_set = pd.read_csv('../results/%s/pattern sets/train/stat_rel.csv' % self.dataset)
        test_set = pd.read_csv('../results/%s/pattern sets/test/stat_rel.csv' % self.dataset)
        train_t_set = pd.read_csv('../results/%s/pattern sets/train/stat_t_rel.csv' % self.dataset)
        test_t_set = pd.read_csv('../results/%s/pattern sets/test/stat_t_rel.csv' % self.dataset)

        for s in ['temporal', '']:
            data_train = train_t_set if s == 'temporal' else train_set
            data_test = test_t_set if s == 'temporal' else test_set
            rel_num_tr = data_train.loc[:, 'number'].sum()
            rel_num_te = data_test.loc[:, 'number'].sum()

            data_train.set_index('relation', inplace=True)
            data_test.set_index('relation', inplace=True)
            fig, ax = plt.subplots(2, 1, figsize=(24, 6), sharex=True)
            ax[0].bar(data_train.index, data_train.loc[:, 'number'],
                      label='number of %s relation in train set' % s, color='#f9766e', edgecolor='grey', alpha=0.5)
            ax[0].bar(data_test.index, data_test.loc[:, 'number'],
                      label='number of %s in test set' % s, color='#00bfc4', edgecolor='grey', alpha=0.5)

            ax[1].bar(data_train.index, data_train.loc[:, 'number'] / rel_num_tr,
                      label='percentage of %s relation in train set' % s, color='#f9766e', edgecolor='grey', alpha=0.5)
            ax[1].bar(data_test.index, data_test.loc[:, 'number'] / rel_num_te,
                      label='percentage of %s in test set' % s, color='#00bfc4', edgecolor='grey', alpha=0.5)

            ax[0].set_xlabel('relations', fontsize=12)
            ax[1].set_xlabel('relations', fontsize=12)
            ax[0].set_ylabel('number', fontsize=12)
            ax[1].set_ylabel('percentage', fontsize=12)
            for i in range(2):
                ax[i].tick_params(axis='x', length=0, rotation=30)
                ax[i].grid(axis='y', alpha=0.5, ls='--')
                ax[i].legend(frameon=False)

            fig.suptitle('({}) {} relation distribution'.format(self.dataset, s))
            plt.xticks([])

            save_path = '../results/{}/statistics/'.format(self.dataset)
            plt.savefig(save_path + '%s_Distribution.png' % s, dpi=300)
            # plt.show()
