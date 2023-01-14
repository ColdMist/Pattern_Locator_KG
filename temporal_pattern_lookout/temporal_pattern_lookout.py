"""
PatternLookout类：处理三元组
TemporalPatternLookout类：PatternLookout的子类，处理四元组
"""


import numpy as np
import pandas as pd
import os


class PatternLookout:
    def __init__(self, temporal=False):
        self.temporal = temporal
        self.num_triples = None
        self.num_self_sym = None
        self.num_symmetric = None
        self.num_inverse = None

        self.intersected = None
        self.diff = None
        self.original = None
        self.reversed = None
        self.concat = None
        self.non_dup_concat = None

    @staticmethod
    def data_loader(dir_name, data_name, file_name):
        read_path = os.path.join(os.path.join(dir_name, data_name), file_name)
        data = pd.read_table(read_path, header=None, names=['head', 'relation', 'tail'])
        # if self.temporal and data.shape[1] >= 5:
        #     data = data.iloc[:, :4]
        return data

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

        self.num_self_sym = num_ss
        self.intersected = pd.merge(non_dup_data, data_reversed, how='inner')
        self.diff = pd.concat([non_dup_data, data_reversed]).drop_duplicates(keep=False)
        self.original = non_dup_data
        self.reversed = data_reversed
        self.concat = pd.concat([non_dup_data, data_reversed], axis=0)
        self.non_dup_concat = self.concat.drop_duplicates()
        return

    def find_symmetric(self):
        assert self.intersected is not None, 'please run "initialize" first'
        set_intersected_ = self.intersected
        num_symm = (len(set_intersected_) - self.num_self_sym) / 2
        assert num_symm % 1 == 0, 'number of symmetric should be "int"'
        self.num_symmetric = num_symm
        return set_intersected_

    def find_reflexive(self):
        assert self.intersected is not None, 'please run "initialize" first'
        ref = self.intersected[self.intersected.iloc[:, 0] == self.intersected.iloc[:, 2]]
        return ref

    def find_inverse(self):
        inv = self.concat.drop_duplicates(subset=['head', 'tail'])
        assert self.non_dup_concat is not None, 'please run "initialize" first'
        inv = pd.concat([inv, self.non_dup_concat]).drop_duplicates(keep=False)
        self.num_inverse = inv.shape[0]
        return inv



patternLooker = PatternLookout(True)
# dataset = patternLooker.data_loader('data', 'FB15K', 'train').iloc[:1000, :]
dataset = pd.DataFrame([[1,2,3], [3,2,1], [3,3,3], [4,5,6], [6,5,4], [7,5,8], [11,2,4], [9,9,5], [7,8,9], [9,10,7]], columns=['head', 'relation', 'tail'])
_ = patternLooker.statistics(dataset)
patternLooker.initialize(dataset)
set_symmetric = patternLooker.find_symmetric()
set_reflexive = patternLooker.find_reflexive()
_ = patternLooker.find_inverse()






print('--------------')
