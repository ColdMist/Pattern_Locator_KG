"""
PatternLookout：for triples
TemporalPatternLookout：subclass of PatternLookout, for quaternion
"""

import numpy as np
import pandas as pd
import os
from scipy.special import comb
import time

class PatternLookout:
    def __init__(self, temporal=False):
        self.temporal = temporal
        self.num_data = None
        self.num_triples = None
        self.num_reflexive = None
        self.num_symmetric = None
        self.num_anti_symmetric = None
        self.num_inverse = None
        self.num_implication = None

        self.intersected = None
        self.diff = None
        self.original = None
        self.reversed = None
        self.concat = None
        self.non_dup_concat = None

    @staticmethod
    def data_loader(dir_name, data_name, file_name, temporal):
        read_path = os.path.join(os.path.join(dir_name, data_name), file_name)
        if not temporal:
            data = pd.read_table(read_path, header=None, names=['head', 'relation', 'tail'], index_col=False)
        else:
            data = pd.read_table(read_path, header=None, names=['head', 'relation', 'tail', 'time'], index_col=False)
        return data

    @ staticmethod
    def count(data):
        dic = data.value_counts(subset=['head', 'tail']).reset_index()
        dic.rename(columns={0: 'num'}, inplace=True)
        dic = dict(zip([(i, j) for i, j in zip(dic['head'], dic['tail'])], dic['num']))
        check = set()
        num = 0
        for paar in dic:
            if paar in check:
                continue
            paar_ = (paar[1], paar[0])
            if paar_ in dic:
                num += dic[paar] * dic[paar_]
                check.add(paar_)
                check.add(paar)
        return num

    def statistics(self, data):
        triples = data.apply(lambda x: tuple(x), axis=1).values.tolist()
        num_triples = len(set(triples))
        self.num_triples = num_triples
        return num_triples

    def initialize(self, data):
        non_dup_data = data.drop_duplicates()
        num_ss = np.sum(non_dup_data.iloc[:, 0] == non_dup_data.iloc[:, 2])
        data_reversed = non_dup_data.copy()

        data_reversed.iloc[:, 0], data_reversed.iloc[:, 2] = data_reversed.iloc[:, 2].copy(), data_reversed.iloc[:, 0].copy()

        self.num_data = int(non_dup_data.shape[0])
        self.num_reflexive = int(num_ss)
        self.intersected = pd.merge(non_dup_data, data_reversed, how='inner')
        self.diff = pd.concat([non_dup_data, data_reversed]).drop_duplicates(keep=False)
        self.original = non_dup_data
        self.reversed = data_reversed
        self.concat = pd.concat([non_dup_data, data_reversed], axis=0)
        self.non_dup_concat = self.concat.drop_duplicates()
        return

    def find_symmetric(self):  # reflexive belongs to symmetric or not
        assert self.intersected is not None, 'please run "initialize" first'
        symm = self.intersected
        anti_symm = pd.concat([self.original, symm]).drop_duplicates(keep=False)
        num_symm = (len(symm) - self.num_reflexive) / 2
        assert num_symm % 1 == 0, 'number of symmetric should be "int"'
        self.num_symmetric = int(num_symm)
        self.num_anti_symmetric = self.num_data - self.num_symmetric
        return symm, anti_symm

    def find_reflexive(self):
        assert self.intersected is not None, 'please run "initialize" first'
        ref = self.intersected[self.intersected.iloc[:, 0] == self.intersected.iloc[:, 2]]
        return ref

    def find_inverse(self):
        assert self.reversed is not None, 'please run "initialize" first'
        assert self.original is not None, 'please run "initialize" first'
        inv = pd.merge(self.original, self.reversed, on=['head', 'tail'], how='inner')
        inv = inv[inv.loc[:, 'relation_x'] != inv.loc[:, 'relation_y']]
        inv.rename(columns={'relation_x': 'relation'}, inplace=True)
        inv = pd.merge(self.original, inv.iloc[:, :3], how='inner').drop_duplicates()
        if self.temporal:
            self.num_inverse = 0
            inv_t = inv.groupby('time')
            for data_t in inv_t:
                self.num_inverse += self.count(data_t[-1])
        else:
            self.num_inverse = self.count(inv)
        self.num_inverse = int(self.num_inverse)
        return inv

    def find_implication(self):
        assert self.original is not None, 'please run "initialize" first'
        imp = self.original.drop_duplicates(subset=['head', 'tail'], keep=False)
        imp = pd.concat([imp, self.original]).drop_duplicates(keep=False)
        imp = imp.drop_duplicates()
        imp_ = imp.groupby(['head', 'tail'])
        self.num_implication = 0
        for group in imp_:
            cnt = comb(group[-1].shape[0], 2)
            self.num_implication += cnt
        self.num_implication = int(self.num_implication)
        return imp


class TemporalPatternLookout(PatternLookout):
    def __init__(self, temporal=True):
        super(TemporalPatternLookout, self).__init__()
        self.temporal = temporal
        self.num_evolve = None
        self.num_t_inverse = None
        self.num_t_relation = None
        self.timeline = None

    def find_evolve(self):
        evo = self.find_implication()
        self.num_evolve = self.num_implication
        return evo

    def find_temporal_inverse(self):
        inv = self.find_inverse()
        self.num_t_inverse = int(self.count(inv))
        return inv

    def find_temporal_relation(self):
        assert self.original is not None, 'please run "initialize" first'
        self.timeline = self.original.drop_duplicates(subset=['time']).loc[:, 'time'].reset_index(drop=True)
        timeline = len(self.timeline)
        r_groups = self.original.groupby(['relation'])
        s_t_rel = set()
        for rel in r_groups:
            if len(rel[-1].drop_duplicates(subset=['time'])) < timeline:
                s_t_rel.add(rel[-1]['relation'].iloc[0])
        self.num_t_relation = len(s_t_rel)
        return s_t_rel


def main():
    print('--------------------Begin--------------------------------------------')
    start = time.time()
    patternlooker = TemporalPatternLookout()
    dataset = patternlooker.data_loader('data', 'ICEWS14_TA', 'train2id.txt', temporal=True).iloc[:, :]

    # initialize
    _ = patternlooker.statistics(dataset)
    patternlooker.initialize(dataset)

    # Static Logical Temporal Patterns
    set_symmetric, set_anti_symmetric = patternlooker.find_symmetric()
    set_reflexive = patternlooker.find_reflexive()
    set_inverse = patternlooker.find_inverse()
    set_implication = patternlooker.find_implication()

    # Dynamic Logical Temporal Patterns
    set_evolve = patternlooker.find_evolve()
    set_t_inverse = patternlooker.find_temporal_inverse()
    set_t_relation = patternlooker.find_temporal_relation()
    end = time.time()
    print('It takes {} seconds'.format(end - start))
    print('Number of symmetric is: {} \n'
          'Number of anti_symmetric is: {} \n'
          'Number of reflexive is: {} \n'
          'Number of inverse is: {} \n'
          'Number of temporal inverse is: {} \n'
          'Number of evolves is: {} \n'
          'Number of temporal relation is: {} \n'.format(patternlooker.num_symmetric, patternlooker.num_anti_symmetric,
                                                         patternlooker.num_reflexive, patternlooker.num_inverse,
                                                         patternlooker.num_t_inverse, patternlooker.num_evolve,
                                                         patternlooker.num_t_relation))
    print('--------------------Finish-------------------------------------------')

if __name__ == '__main__':
    main()
