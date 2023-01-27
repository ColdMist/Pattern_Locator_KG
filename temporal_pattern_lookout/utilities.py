# -*- coding: utf-8 -*- 
# @Time : 2023/1/24 19:24 
# @Author : Yinan 
# @File : utilities.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticke
import numpy as np
# a) in training set: freq of symmetric grounding per relation
# b) total freq of samples for each relations
# c) compute percentage
# d) distubution of training and teat set
# e) put a threshold to select symmetric etc
# f) for static symmetric: count how many samples by that relation exist in test
# another: for dynamic symmetric: select top k dynamic symmetric relations based on trainig you already got,
# then count how many of test samples their Premise exist in train  ,premise->conclusion


class AnalysisTools:
    def __init__(self):
        pass

    def occurance_symmetric(self, symm_set, stat_rel):
        symm_set = pd.DataFrame(symm_set.value_counts('relation')).rename(columns={0: 'number'})
        stat_rel = stat_rel.set_index('relation', inplace=False)
        freq = (symm_set / stat_rel).fillna(0).reset_index()
        return freq.sort_values(by='number', ascending=False)


    # def occurance_temporal_relation(self, symm_set, stat_rel):
    #     symm_set = pd.DataFrame(symm_set.value_counts('relation')).rename(columns={0: 'freq'})
    #     stat_rel.set_index('relation', inplace=True)
    #     freq = (symm_set / stat_rel).fillna(0).reset_index()
    #     return freq.sort_values(by='freq', ascending=False)
    #


class PlotTools:
    def __init__(self):
        pass

    def plot_distribution_rel(self, train_set, test_set, showall=False, dynamic=False, sparse=False):
        if not showall:
            train_set = train_set[~train_set['freq'].isin([0])]
            test_set = test_set[~test_set['freq'].isin([0])]
        train_set.reset_index(inplace=True)
        test_set.reset_index(inplace=True)
        fig, ax = plt.subplots(1, 1)
        ax.bar(train_set.loc[:, 'relation'].apply(str), train_set.loc[:, 'freq'] / train_set.loc[:, 'freq'].sum(), label='Train', color='#f9766e', edgecolor='grey', alpha=0.5)
        ax.bar(test_set.loc[:, 'relation'].apply(str), test_set.loc[:, 'freq'] / test_set.loc[:, 'freq'].sum(), label='Test', color='#00bfc4', edgecolor='grey', alpha=0.5)
        ax.set_xlabel('relations', fontsize=12)
        ax.set_ylabel('percentage', fontsize=12)
        ax.tick_params(axis='x', length=0, rotation=30)
        ax.grid(axis='y', alpha=0.5, ls='--')
        ax.legend(frameon=False)
        ax.set_title('Dynamic' if dynamic else 'Static')

        if sparse:
            ax.xaxis.set_major_locator(ticke.MultipleLocator(base=5))
        plt.show()
