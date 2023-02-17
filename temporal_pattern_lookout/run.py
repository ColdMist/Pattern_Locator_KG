# -*- coding: utf-8 -*- 
# @Time : 2023/2/7 14:27 
# @Author : Yinan 
# @File : run.py

import argparse
import temporal_pattern_lookout
import evaluation

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='selfmade', choices=['icews14', 'wikidata_TA', 'selfmade'], help='Knowledge graph dataset')
parser.add_argument('--pattern', default='evolve'
                    , choices=['symmetric', 'temporal symmetric', 'inverse'
        , 'temporal inverse', 'implication', 'temporal implication', 'evolve'])
args = parser.parse_args()

temporal_pattern_lookout.main(args.dataset)
evaluation.main(args.dataset, args.pattern)