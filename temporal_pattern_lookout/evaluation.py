# -*- coding: utf-8 -*- 
# @Time : 2023/1/27 15:00 
# @Author : Yinan 
# @File : evaluation.py
import os

import pandas as pd
from utilities import PlotTools


def plot_sysm_distribution(on='symmetric'):
    stat_path = '../results/ICEWS14_TA/statistics'
    temp_path = stat_path + '/dynamic/{}/'.format(on)
    static_path = stat_path + '/static/{}/'.format(on)

    ploter = PlotTools()
    for file_ord in [temp_path, static_path]:
        data_path = list()
        for file in os.listdir(file_ord):
            data_path.append(file)

        test, train = pd.read_csv(file_ord + data_path[0], index_col=0), pd.read_csv(file_ord + data_path[1], index_col=0)
        ploter.plot_distribution_rel(train, test, showall=False
                                     , dynamic=True if file_ord == temp_path else False
                                     , sparse=True if on == 'relations' else False)

for on in ['symmetric', 'relations']:
    plot_sysm_distribution(on=on)