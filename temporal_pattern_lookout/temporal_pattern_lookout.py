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

    @staticmethod
    def data_loader(dir_name, data_name, file_name):
        read_path = os.path.join(os.path.join(dir_name, data_name), file_name)
        data = pd.read_table(read_path, header=None)
        # if self.temporal and data.shape[1] >= 5:
        #     data = data.iloc[:, :4]
        return data

    def statistics(self, data):
        triples = data.apply(lambda x: tuple(x), axis=1).values.tolist()
        num_triples = len(set(triples))
        self.num_triples = num_triples
        return num_triples

    def generate_reverse_data(self, data):
        non_dup_data = data.drop_duplicates()
        num_ss = np.sum(non_dup_data.iloc[:, 0] == non_dup_data.iloc[:, 2])
        data_reversed = non_dup_data.copy()
        temp = data_reversed.iloc[:, 0].copy()
        data_reversed.iloc[:, 0] = data_reversed.iloc[:, 2]
        data_reversed.iloc[:, 2] = temp
        # c_data = pd.concat([non_dup_data, data_reversed], axis=0)
        self.num_self_sym = num_ss
        return non_dup_data, data_reversed, num_ss

    def find_symmetric(self, _data, data_):
        set_intersected_ = pd.merge(_data, data_, how='inner')
        set_diff_ = pd.concat([_data, data_]).drop_duplicates(keep=False)
        num_symm = (len(set_intersected_) - self.num_self_sym) / 2
        assert num_symm % 1 == 0, 'number of symmetric should be "int"'
        self.num_symmetric = num_symm
        return set_intersected_

    def find_reflexive(self):
        pass


patternLooker = PatternLookout(True)
dataset = patternLooker.data_loader('data', 'FB15K', 'train')
# dataset = pd.DataFrame([[1,2,3],[3,2,1],[3,3,3],[4,5,6],[6,5,4],[7,5,8],[11,2,4],[9,9,5]])
_ = patternLooker.statistics(dataset)
f_data, b_data, num_self_sym = patternLooker.generate_reverse_data(dataset)
set_symmetric = patternLooker.find_symmetric(f_data, b_data)

# find symmetric





print('--------------')
