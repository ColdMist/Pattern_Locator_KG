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

# TODO: 3. make (ri, rj) inverse dict, count for each pair: 2*#(ri, rj) / (#ri + #rj) on train
# TODO: 4. for test, for certain ri, looking up dict (for a (ri, rj) regarding a threshold)
#  , check if (. rj .) exists in train, if yes, count 1
# TODO: 5. make (ri --> rj) implication dict, count 1 each pair: #(ri --> rj) / (#ri) on train
# TODO: 6. check for (ri) in test set, looking up dict(rj-->ri)(premise --> conclusion) and look if (. rj .) in train



def plot_distribution(dataset, pattern):
    for t in ['train', 'test']:
        typ = 'dynamic' if (pattern == 'evolve' or pattern.split()[0] == 'temporal') else 'static'
        data = pd.read_csv('../results/{}/statistics/{}/{}/freq_{}.csv'.format(dataset, t, typ, pattern))
        fig, ax = plt.subplots(2, 1, figsize=(24, 6), sharex=True)

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

        fig.suptitle('({}) {}-{}-{}'.format(dataset, t, typ, pattern))
        plt.xticks([])

        save_path = '../results/{}/statistics/{}/{}/'.format(dataset, t, typ)
        plt.savefig(save_path + '%s_Distribution.png' % pattern, dpi=300)
        # plt.show()


def main(dataset, pattern):
    analysis = AnalysisTools(dataset, pattern)
    analysis.pattern_analyse()
    # analysis.conclusion_premise_paar()
    analysis.pattern_pair_analyse()

    # ptool = PlotTools(dataset)
    # ptool.plot_train_test_distribution()
    # plot_distribution(dataset, pattern)
