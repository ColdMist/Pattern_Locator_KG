import os
import argparse
from utils.utility_functions import *

def main(args):
    data_dir = args.data_dir
    data_name = args.data_name
    file_name = args.file_name
    save_pattern_dir = args.save_pattern_dir

    save_pattern_path = os.path.join(
        os.path.join(data_dir, data_name), save_pattern_dir
    )
    file_path = os.path.join(os.path.join(data_dir, data_name), file_name)
    all_triple_df = pd.read_table(file_path, header=None, dtype=str)
    triples, observed_triples = build_indexed_dictioary(all_triple_df)

    symmetric_patterns, count_sym = find_symmetric(
        triples, observed_triples=observed_triples
    )
    reflexive_patterns, count_ref = find_reflexive(all_triple_df)
    implication_patterns, count_imp = find_implication(
        all_triple_df, triples, observed_triples
    )
    inverse_patterns, count_inv = find_inverse(all_triple_df, triples, observed_triples)
    transitive_patterns, count_tran = find_transitive(
        all_triple_df, triples, observed_triples
    )
    composition_patterns, count_tran = find_composition(
        all_triple_df, triples, observed_triples
    )

    if not os.path.exists(save_pattern_path):
        # If not, create it
        os.makedirs(save_pattern_path)

    if symmetric_patterns!=None:
        pd.DataFrame(symmetric_patterns).to_csv(
            os.path.join(save_pattern_path, "symmetric_patterns.csv"),
            sep="\t",
            header=None,
            index=False,
        )
    if reflexive_patterns!=None:
        pd.DataFrame(reflexive_patterns).to_csv(
            os.path.join(save_pattern_path, "reflexive_patterns.csv"),
            sep="\t",
            header=None,
            index=False,
        )
    if implication_patterns!=None:
        pd.DataFrame(implication_patterns).to_csv(
            os.path.join(save_pattern_path, "implication_patterns.csv"),
            sep="\t",
            header=None,
            index=False,
        )
    if inverse_patterns!=None:
        pd.DataFrame(inverse_patterns).to_csv(
            os.path.join(save_pattern_path, "inverse_patterns.csv"),
            sep="\t",
            header=None,
            index=False,
        )
    if transitive_patterns!=None:
        pd.DataFrame(transitive_patterns).to_csv(
            os.path.join(save_pattern_path, "transitive_patterns.csv"),
            sep="\t",
            header=None,
            index=False,
        )
    if composition_patterns!=None:
        pd.DataFrame(composition_patterns).to_csv(
            os.path.join(save_pattern_path, "composition_patterns.csv"),
            sep="\t",
            header=None,
            index=False,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', required=True, help='data directory')
    parser.add_argument('--data_name', required=True, help='data name')
    parser.add_argument('--file_name', required=True, help='file name')
    parser.add_argument('--save_pattern_dir', required=True, help='save pattern directory')

    args = parser.parse_args()

    main(args)
