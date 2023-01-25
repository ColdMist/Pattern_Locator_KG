# -*- coding: utf-8 -*- 
# @Time : 2023/1/24 19:24 
# @Author : Yinan 
# @File : utilities.py


# a) in training set: number of symmetric grounding per relation
# b) total number of samples for each relations
# c) compute percentage
# d) distubution of training and teat set
# e) put a threshold to select symmetric etc
# f) for static symmetric: count how many samples by that relation exist in test
# another: for dynamic symmetric: select top k dynamic symmetric relations based on trainig you already got,
# then count how many of test samples their Premise exist in train  ,premise->conclusion


class AnalysisTools():
    def __init__(self):
        pass

    def cal_num_symmetric(self, symm_set):
        symm_dict = symm_set.value_counts('relation').to_dict()
        return symm_dict

    def num_each_relations(self, df):
        num_rel = df.value_counts('relation').to_dict()
        return num_rel

    def cal_percentage(self, df1, df2):
        percentage = df1.shape[0] / df2.shape[0] * 100
        return '{.3f}%'.format(percentage)



