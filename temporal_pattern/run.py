# -*- coding: utf-8 -*- 
# @Time : 2023/2/7 14:27 
# @Author : Yinan 
# @File : run.py

import argparse

import evaluation
import temporal_pattern_lookout
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='icews14', choices=['icews14', 'wikidata_TA', 'icews15'], help='Knowledge graph dataset')
# parser.add_argument('--pattern', default='evolve'
#                     , choices=['symmetric', 'temporal symmetric', 'inverse'
#         , 'temporal inverse', 'implication', 'temporal implication', 'evolve'])
parser.add_argument('--threshold', default=0.5, type=float)
args = parser.parse_args()

temporal_pattern_lookout.main(args.dataset)

for p in ['symmetric', 'temporal symmetric', 'inverse'
        , 'temporal inverse', 'implication', 'temporal implication', 'evolve']:
    evaluation.main(args.dataset, p, args.threshold)

