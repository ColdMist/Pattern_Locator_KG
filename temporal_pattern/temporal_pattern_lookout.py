# -*- coding: utf-8 -*- 
# @Time : 2023/2/17 21:10 
# @Author : Yinan 
# @File : temporal_pattern.py

import numpy as np
import pandas as pd
import os
from scipy.special import comb
import time


class TemporalPatternLookout:
    def __init__(self):
        super(TemporalPatternLookout, self).__init__()
        self.temporal = True
        self.num_t_data = None
        self.num_quaternions = None
        self.num_t_reflexive = None
        self.num_symmetric = None
        self.num_t_symmetric = None
        # self.anti_num_symmetric = None
        # self.num_t_anti_symmetric = None
        self.num_evolve = None
        self.num_inverse = None
        self.num_t_inverse = None
        self.num_implication = None
        self.num_t_implication = None
        self.num_t_relations = None
        self.timeline = None
        self.stat_t_rel = None

        self.intersected = None
        self.f3_intersected = None
        self.comp = None
        self.original = None
        self.reversed = None
        self.concat = None

    @staticmethod
    def count(data):
        dic = data.value_counts(subset=['head', 'relation', 'tail']).reset_index()
        dic.rename(columns={0: 'num'}, inplace=True)
        dic = dict(zip([(i, j, k) for i, j, k in zip(dic['head'], dic['relation'], dic['tail'])], dic['num']))
        check = set()
        num = 0
        for paar in dic:
            check.add(paar)
            for paar_ in dic:
                if paar_ in check:
                    continue
                elif paar_[0] == paar[-1] and paar_[-1] == paar[0] and paar_[1] != paar[1]:
                    num += dic[paar] * dic[paar_]

        return num

    @staticmethod
    def count_comb(data, temporal=False):
        cnt = 0
        if temporal:
            data = data.groupby('time')
            for _, dt in data:
                ht_data = dt.groupby(['head', 'tail'])
                for _, ht in ht_data:
                    cnt += comb(ht.shape[0], 2)
        else:
            ht_data = data.groupby(['head', 'tail'])
            for _, ht in ht_data:
                cnt += comb(ht.shape[0], 2)
                # if temporal:
                #     ht_vc = ht.value_counts('relation')
                #     for c in ht_vc[ht_vc > 1].values:
                #         cnt -= comb(c, 2)
        return cnt

    def data_loader(self, dir_name, data_name, file_name):
        read_path = os.path.join(os.path.join(dir_name, data_name), file_name)
        if not self.temporal:
            data = pd.read_table(read_path, header=None, names=['head', 'relation', 'tail'], index_col=False)
        else:
            data = pd.read_table(read_path, header=None, names=['head', 'relation', 'tail', 'time'], index_col=False)
        return data

    def initialize(self, data):
        non_dup_data = data.drop_duplicates()
        num_ss = np.sum(non_dup_data.iloc[:, 0] == non_dup_data.iloc[:, 2])
        data_reversed = non_dup_data.copy()

        data_reversed.iloc[:, 0], data_reversed.iloc[:, 2] = data_reversed.iloc[:, 2].copy(), data_reversed.iloc[:,0].copy()
        self.num_quaternions = non_dup_data.shape[0]
        self.original = non_dup_data
        self.reversed = data_reversed

        self.num_t_data = int(self.original.shape[0])
        self.num_t_reflexive = int(num_ss)

        self.intersected = pd.merge(self.original, self.reversed, how='inner')
        self.f3_intersected = pd.merge(self.original, self.reversed, how='inner', on=['head', 'relation', 'tail'])

        self.concat = pd.concat([self.original, self.reversed], axis=0)

        self.timeline = self.original.drop_duplicates(subset=['time']).loc[:, 'time'].reset_index(drop=True)

        stat = pd.DataFrame(data.value_counts('relation')).reset_index().rename(columns={0: 'number'})
        self.stat_t_rel = stat

    def find_reflexive(self):
        assert self.intersected is not None, 'please run "initialize" first'
        ref = self.intersected[self.intersected.iloc[:, 0] == self.intersected.iloc[:, 2]]
        return ref

    def find_symmetric(self):
        assert self.intersected is not None, 'please run "initialize" first'
        symm = self.f3_intersected
        num_symm = (len(symm) + self.num_t_reflexive) / 2
        symm.rename(columns={'time_x': 'time'}, inplace=True)
        symm = pd.merge(self.original, symm.iloc[:, :4], how='inner').drop_duplicates()
        assert num_symm % 1 == 0, 'number of symmetric should be "int"'
        self.num_symmetric = int(num_symm)
        return symm

    def find_temporal_symmetric(self):
        assert self.intersected is not None, 'please run "initialize" first'
        symm = self.intersected
        num_symm = (len(symm) + self.num_t_reflexive) / 2
        assert num_symm % 1 == 0, 'number of symmetric should be "int"'
        self.num_t_symmetric = int(num_symm)
        # self.num_t_anti_symmetric = self.num_t_data - self.num_t_symmetric
        return symm

    def find_inverse(self):
        assert self.reversed is not None, 'please run "initialize" first'
        assert self.original is not None, 'please run "initialize" first'
        inv = pd.merge(self.original, self.reversed, on=['head', 'tail'], how='inner')
        inv = inv[inv.loc[:, 'relation_x'] != inv.loc[:, 'relation_y']]
        inv.rename(columns={'relation_x': 'relation'}, inplace=True)
        inv.rename(columns={'time_x': 'time'}, inplace=True)
        inv = pd.merge(self.original, inv.iloc[:, :4], how='inner').drop_duplicates()
        self.num_inverse = self.count(inv)
        self.num_inverse = int(self.num_inverse)
        return inv

    def find_temporal_inverse(self):
        assert self.reversed is not None, 'please run "initialize" first'
        assert self.original is not None, 'please run "initialize" first'
        inv = pd.merge(self.original, self.reversed, on=['head', 'tail', 'time'], how='inner')
        inv = inv[inv.loc[:, 'relation_x'] != inv.loc[:, 'relation_y']]
        inv.rename(columns={'relation_x': 'relation'}, inplace=True)
        inv.drop('relation_y', inplace=True, axis=1)
        inv = pd.merge(self.original, inv, how='inner').drop_duplicates()
        num_t_inverse = 0
        set_inv = pd.DataFrame()
        inv_t = inv.groupby('time')
        for _, data_t in inv_t:
            cnt = self.count(data_t)
            num_t_inverse += cnt
            if cnt:
                set_inv = pd.concat([set_inv, data_t], axis=0)
        self.num_t_inverse = int(num_t_inverse)
        return set_inv

    def find_implication(self):
        assert self.original is not None, 'please run "initialize" first'
        imp = self.original.drop_duplicates(subset=['head', 'tail'], keep=False)
        imp = pd.concat([imp, self.original]).drop_duplicates(keep=False)
        self.num_implication = self.count_comb(imp)
        self.num_implication = int(self.num_implication)
        return imp

    def find_temporal_implication(self):
        assert self.original is not None, 'please run "initialize" first'
        imp = self.original.drop_duplicates(subset=['head', 'tail', 'time'], keep=False)
        imp = pd.concat([imp, self.original]).drop_duplicates(keep=False)
        self.num_t_implication = int(self.count_comb(imp, True))
        return imp

    def find_evolve(self):
        def cal_evolve(df: pd.DataFrame):
            cnt = 0
            cnt_overlap = 0
            for index, row in df.iterrows():
                row = pd.DataFrame(row).T
                r = row.loc[index, 'relation']
                t = row.loc[index, 'time']
                temp = df[(df['relation'] != r) & (df['time'] >= t)]

                if temp.shape[0] == 0:
                    df.drop(index, axis=0)
                    continue
                cnt += temp.shape[0]
            return df, cnt

        assert self.original is not None, 'please run "initialize" first'
        evo = self.original.drop_duplicates(subset=['head', 'tail'], keep=False)
        evo = pd.concat([evo, self.original]).drop_duplicates(keep=False)
        set_evo = pd.DataFrame()
        evo = evo.groupby(['head', 'tail'])
        num_evo = 0
        for _, data_e in evo:
            e, c = cal_evolve(data_e)
            num_evo += c
            set_evo = pd.concat([set_evo, e])
        self.num_evolve = int(num_evo)
        return set_evo

    def find_temporal_relation(self):
        assert self.original is not None, 'please run "initialize" first'
        timeline = len(self.timeline)
        s_t_rel = set()
        ht_groups = self.original.groupby(['head', 'tail'])
        for _, ht in ht_groups:
            r_groups = ht.groupby(['relation'])
            for _, rel in r_groups:
                if len(rel.drop_duplicates(subset=['time'])) < timeline:
                    s_t_rel.add(rel['relation'].iloc[0])
        self.num_t_relations = len(s_t_rel)
        return s_t_rel


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(dataname):
    print('-----------------Data Reprocess Begin--------------------------------------------')
    start = time.time()

    for data in ['train2id.txt', 'test2id.txt']:
        patternlooker = TemporalPatternLookout()
        dataset = patternlooker.data_loader('data', dataname, data).iloc[:, :]

        patternlooker.initialize(dataset)

        # Static Logical Temporal Patterns
        set_symmetric = patternlooker.find_symmetric()

        set_reflexive = patternlooker.find_reflexive()
        set_inverse = patternlooker.find_inverse()
        set_implication = patternlooker.find_implication()

        # Dynamic Logical Temporal Patterns
        set_t_symmetric = patternlooker.find_temporal_symmetric()
        set_evolve = patternlooker.find_evolve()
        set_t_implication = patternlooker.find_temporal_implication()
        set_t_inverse = patternlooker.find_temporal_inverse()
        set_t_relation = patternlooker.find_temporal_relation()
        end = time.time()

        stat_dict = {
            'number of static symmetric': patternlooker.num_symmetric
            , 'number of static inverse': patternlooker.num_inverse
            , 'number of implication': patternlooker.num_implication

            , 'number of temporal data': patternlooker.num_t_data
            , 'number of temporal relations': patternlooker.num_t_relations
            , 'number of quaternions': patternlooker.num_quaternions
            , 'number of temporal symmetric': patternlooker.num_t_symmetric
            , 'number of temporal inverse': patternlooker.num_t_inverse
            , 'number of reflexive': patternlooker.num_t_reflexive
            , 'number of temporal implication': patternlooker.num_t_implication
            , 'number of evolve': patternlooker.num_evolve}

        save_path = '../results/{}/statistics'.format(dataname)
        makedir(save_path)
        with open(save_path + '/stats_{}.txt'.format(data[:-7]), 'w') as file:
            file.write(str(stat_dict))

        # save sets
        set_path = '../results/{}/pattern sets/{}'.format(dataname, data[:-7])
        makedir(set_path)

        set_reflexive.to_csv(set_path + '/set reflexive.csv', index=False)
        set_symmetric.to_csv(set_path + '/set symmetric.csv', index=False)
        set_inverse.to_csv(set_path + '/set inverse.csv', index=False)
        set_implication.to_csv(set_path + '/set implication.csv', index=False)
        set_t_inverse.to_csv(set_path + '/set temporal inverse.csv', index=False)
        set_t_symmetric.to_csv(set_path + '/set temporal symmetric.csv', index=False)
        set_t_implication.to_csv(set_path + '/set temporal implication.csv', index=False )
        set_evolve.to_csv(set_path + '/set evolve.csv', index=False)
        pd.DataFrame(set_t_relation).to_csv(set_path + '/set temporal relation.csv', index=False)
        patternlooker.stat_t_rel.to_csv(set_path + '/stat_t_rel.csv', index=False)

        print('It takes {} seconds on {}'.format(end - start, data))
    #     print('Number of symmetric is: {} \n'
    #           'Number of anti_symmetric is: {} \n'
    #           'Number of reflexive is: {} \n'
    #           'Number of inverse is: {} \n'
    #           'Number of temporal inverse is: {} \n'
    #           'Number of temporal implication is: {} \n'
    #           'Number of evolves is: {} \n'
    #           'Number of temporal relation is: {} \n'.format(patternlooker.num_symmetric, patternlooker.num_anti_symmetric,
    #                                                          patternlooker.num_reflexive, patternlooker.num_inverse,
    #                                                          patternlooker.num_t_inverse, patternlooker.num_implication,
    #                                                          patternlooker.num_evolve, patternlooker.num_t_relations))
    print('----------------Data Reprocess Finish-------------------------------------------')