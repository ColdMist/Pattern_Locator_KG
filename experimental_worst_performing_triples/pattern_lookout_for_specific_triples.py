import numpy as np
import pandas as pd
import os

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 12)

def find_patterns_for_triples(triples, pattern_list):
    for index in triples.index:
        row_subject = triples.loc[index,0]
        row_object = triples.loc[index,2]
        #print(pattern_list)
        #cols = list(pattern_list.columns)
        #print(cols)
        filter_row_subject = pattern_list[pattern_list.eq(row_subject).any(1)].reset_index(drop=True)
        #print(filter_row_subject)
        filter_row_subject_object = filter_row_subject[filter_row_subject.eq(row_object).any()]
        #print(filter_row_subject_object)
        print('any pattern does not match this low performing static')


if __name__ == '__main__':
    data_dir = '../data'
    data_name = 'umls'
    worst_performing_triple_file_name = 'low_score_triples_prev.csv'

    pattern_data = pd.read_table('/home/coldmist/PycharmProjects/Pattern_Locator_KG/data/umls/found_patterns/transitive_patterns.csv', header=None, sep='\t')
    low_score_triples = pd.read_table('/home/coldmist/PycharmProjects/Pattern_Locator_KG/data/umls/low_score_triples_prev.csv', header=None, sep='\t')

    find_patterns_for_triples(low_score_triples, pattern_data)
