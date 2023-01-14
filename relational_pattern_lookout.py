import numpy as np
import pandas as pd
import os

def build_indexed_dictioary(data):
    triples = list(zip([h for h in data[0]],
                                [r for r in data[1]],
                                [t for t in data[2]]))
    observed_triples = {}
    for triple in triples:
        observed_triples[triple[0], triple[1], triple[2]] = 1
    return triples, observed_triples

def find_reflexive(data):
    reflexive_triples = data.loc[data[0] == data[2]]
    reflexive_triples = pd.DataFrame(np.array(reflexive_triples), columns=['head', 'relation', 'tail'])
    unique_relations_reflexive_triples = np.array(list(set(list(reflexive_triples['relation']))))

    reflexive_triple_per_relation_count = []
    for unique_relations_reflexive_triple in unique_relations_reflexive_triples:
        # print(unique_relations_reflexive_triple)
        place_holder_relation_triple = reflexive_triples.loc[reflexive_triples['relation'] == unique_relations_reflexive_triple]
        place_holder_relation_triple = np.array(place_holder_relation_triple)
        # print(unique_relations_reflexive_triple, ' : ', len(place_holder_relation_triple))
        stat_count_per_relation = np.array([unique_relations_reflexive_triple, int(len(place_holder_relation_triple))])
        reflexive_triple_per_relation_count.append(stat_count_per_relation)

    reflexive_triple_per_relation = pd.DataFrame(np.array(reflexive_triple_per_relation_count))
    #reflexive_triple_per_relation['reflexive_relation_count'] = reflexive_triple_per_relation['reflexive_relation_count'].astype(str).astype(int)
    #reflexive_triples_per_relation_sorted = reflexive_triple_per_relation.sort_values(by='reflexive_relation_count', ascending=False)

    return reflexive_triples, reflexive_triple_per_relation

def find_implication(triples_df,  triples, observed_triples):
    all_relations = list(set(list(triples_df[1])))
    premise_list = []
    conclusion_list = []
    implication_patterns = []
    #count = 0
    # for implication
    for triple in triples:
        #print(count)
        for r2 in all_relations:
            if (r2 != triple[1]):
                try:
                    exists = observed_triples[triple[0], r2, triple[2]]
                    premise = np.array([triple[0], triple[1], triple[2]])
                    conclusion = np.array([triple[0], r2, triple[2]])
                    #print('premise ', premise)
                    #print('conclusion ', conclusion)
                    #print('######################################')
                    premise_list.append(np.array(premise))
                    conclusion_list.append(np.array(conclusion))
                    implication_pattern = [triple[0],triple[1],triple[2],'->',triple[0],r2,triple[2]]
                    implication_patterns.append(implication_pattern)
                except:
                    continue
        #count += 1

    premise_df = pd.DataFrame(np.array(premise_list)).applymap(str)
    conclusion_df = pd.DataFrame(np.array(conclusion_list)).applymap(str)
    premise_df_unique_relation = np.array(list(set(list(premise_df[1]))))
    conclusion_df_unique_relation = np.array(list(set(list(conclusion_df[1]))))

    implication_triple_per_relation = []
    for premise in premise_df_unique_relation:
        implication_array_place_holder = premise_df.loc[premise_df[1] == premise]
        implication_array_place_holder = np.array(implication_array_place_holder)
        # print(unique_relations_reflexive_triple, ' : ', len(place_holder_relation_triple))
        stat_count_per_relation_implication = np.array(
            [premise, int(len(implication_array_place_holder))])
        implication_triple_per_relation.append(stat_count_per_relation_implication)

    implication_triple_per_relation = pd.DataFrame(np.array(implication_triple_per_relation))
    #implication_triple_per_relation[1] = implication_triple_per_relation[1].astype(str).astype(int)
    #implication_triple_per_relation_sorted = implication_triple_per_relation.sort_values(by=1, ascending=False)


    return implication_patterns, implication_triple_per_relation

def find_inverse(triples_df,  triples, observed_triples):
    #triples_df = pd.DataFrame(triples)
    all_relations = list(set(list(triples_df[1])))
    premise_list = []
    conclusion_list = []
    inverse_patterns = []
    inverse_relations = []

    for triple in triples:
        for r2 in all_relations:
            if (r2 != triple[1]):
                try:
                    exists = observed_triples[triple[2], r2, triple[0]]
                    premise = np.array([triple[0], triple[1], triple[2]])
                    conclusion = np.array([triple[2], r2, triple[0]])
                    #print('premise ', premise)
                    #print('conclusion ', conclusion)
                    #print('######################################')
                    premise_list.append(np.array(premise))
                    conclusion_list.append(np.array(conclusion))
                    inverse_pattern = [triple[0], triple[1], triple[2], '->', triple[2], r2, triple[0]]
                    inverse_patterns.append(inverse_pattern)
                except:
                    continue
    premise_df = pd.DataFrame(np.array(premise_list)).applymap(str)
    conclusion_df = pd.DataFrame(np.array(conclusion_list)).applymap(str)
    premise_df_unique_relation = np.array(list(set(list(premise_df[1]))))
    conclusion_df_unique_relation = np.array(list(set(list(conclusion_df[1]))))

    inverse_triple_per_relation = []
    for premise in premise_df_unique_relation:
        inverse_array_place_holder = premise_df.loc[premise_df[1] == premise]
        inverse_array_place_holder = np.array(inverse_array_place_holder)
        # print(unique_relations_reflexive_triple, ' : ', len(place_holder_relation_triple))
        stat_count_per_relation_inverse = np.array(
            [premise, int(len(inverse_array_place_holder))])
        inverse_triple_per_relation.append(stat_count_per_relation_inverse)

    inverse_triple_per_relation = pd.DataFrame(np.array(inverse_triple_per_relation))
    inverse_triple_per_relation[1] = inverse_triple_per_relation[1].astype(str).astype(int)

    return inverse_patterns, inverse_triple_per_relation

def find_symmetric(triples, observed_triples):
    triples_df = pd.DataFrame(triples)
    premise_list = []
    conclusion_list = []
    symmetric_patterns = []
    symmetric_relations = []
    for triple in triples:
        #print(triple)
        try:
            exists = observed_triples[triple[2], triple[1], triple[0]]
            #print('original_triple ', triple)
            # symmetric = np.array([triple[2], triple[1], triple[0]])
            premise = np.array([triple[0], triple[1], triple[2]])
            conclusion = np.array([triple[2], triple[1], triple[0]])
            #print('premise ', premise)
            #print('conclusion ', conclusion)
            symmtetric_pattern = [triple[0],triple[1],triple[2],'->',triple[2],triple[1],triple[0]]
            symmetric_patterns.append(symmtetric_pattern)
            symmetric_relations.append(triple[1])
            #print('######################################')
            premise_list.append(np.array(premise))
            conclusion_list.append(np.array(conclusion))
        except:
            continue

    premise_df = pd.DataFrame(np.array(premise_list)).applymap(str)
    conclusion_df = pd.DataFrame(np.array(conclusion_list)).applymap(str)
    premise_df_unique_relation = np.array(list(set(list(premise_df[1]))))
    conclusion_df_unique_relation = np.array(list(set(list(conclusion_df[1]))))
    #####################################################################################################
    symmetric_triple_per_relation = []
    for premise in premise_df_unique_relation:
        symmetric_array_place_holder = premise_df.loc[premise_df[1] == premise]
        symmetric_array_place_holder = np.array(symmetric_array_place_holder)
        # print(unique_relations_reflexive_triple, ' : ', len(place_holder_relation_triple))
        stat_count_per_relation_symmetric = np.array(
            [premise, int(len(symmetric_array_place_holder))])
        symmetric_triple_per_relation.append(stat_count_per_relation_symmetric)

    symmetric_triple_per_relation = pd.DataFrame(np.array(symmetric_triple_per_relation))
    symmetric_triple_per_relation[1] = symmetric_triple_per_relation[1].astype(str).astype(int)
    symmetric_triple_per_relation_sorted = symmetric_triple_per_relation.sort_values(by=1, ascending=False)

    return symmetric_patterns, symmetric_triple_per_relation_sorted

def find_transitive(triples_df, triples, observed_triples):
    all_relations = list(set(list(triples_df[1])))
    triples_df.columns = ['h', 'r', 't']
    premise_list = []
    conclusion_list = []
    transitive_patterns = []
    transitive_relations = []
    merged_df = pd.merge(triples_df, triples_df, left_on=['r', 't'], right_on=['r', 'h'], how='inner')
    merged_data = merged_df.to_numpy()
    for triple in merged_data:
        try:
            exists = observed_triples[triple[0], triple[1], triple[4]]
            premise = np.array([triple[0], triple[1], triple[2], triple[3], triple[1], triple[4]])
            conclusion = np.array([triple[0], triple[1], triple[4]])
            transitive_pattern = [triple[0], triple[1], triple[2], 'and', triple[3], triple[1], triple[4], '->',
                                  triple[0], triple[1], triple[4]]
            transitive_patterns.append(transitive_pattern)
            transitive_relations.append(triple[1])
            premise_list.append(np.array(premise))
            conclusion_list.append(np.array(conclusion))
        except:
            continue

    premise_df = pd.DataFrame(np.array(premise_list)).applymap(str)
    conclusion_df = pd.DataFrame(np.array(conclusion_list)).applymap(str)
    premise_df_unique_relation = np.array(list(set(list(premise_df[1]))))
    conclusion_df_unique_relation = np.array(list(set(list(conclusion_df[1]))))
    #####################################################################################################
    transitive_triple_per_relation = []
    for premise in premise_df_unique_relation:
        transitive_array_place_holder = premise_df.loc[premise_df[1] == premise]
        transitive_array_place_holder = np.array(transitive_array_place_holder)
        # print(unique_relations_reflexive_triple, ' : ', len(place_holder_relation_triple))
        stat_count_per_relation_transitive = np.array(
            [premise, int(len(transitive_array_place_holder))])
        transitive_triple_per_relation.append(stat_count_per_relation_transitive)

    transitive_triple_per_relation = pd.DataFrame(np.array(transitive_triple_per_relation))
    transitive_triple_per_relation[1] = transitive_triple_per_relation[1].astype(str).astype(int)
    transitive_triple_per_relation_sorted = transitive_triple_per_relation.sort_values(by=1, ascending=False)

    return transitive_patterns, transitive_triple_per_relation_sorted

if __name__ == '__main__':
    data_dir = 'data'
    data_name = 'FB15K'
    file_name = 'train'
    save_pattern_dir = 'found_patterns'

    save_pattern_path = os.path.join(os.path.join(data_dir,data_name), save_pattern_dir)
    file_path = os.path.join(os.path.join(data_dir,data_name),file_name)
    all_triple_df= pd.read_table(file_path, header=None, dtype=str)
    all_triple_df = all_triple_df.iloc[:1000,:]
    # all_triple_df = pd.DataFrame([[1,2,3],[3,2,1],[3,3,3],[4,5,6],[6,5,4],[7,5,8],[11,2,4],[9,9,5]])
    triples, observed_triples = build_indexed_dictioary(all_triple_df)

    # symmetric_patterns, count_sym = find_symmetric(triples, observed_triples=observed_triples)
    # reflexive_patterns, count_ref = find_reflexive(all_triple_df)
    # implication_patterns, count_imp = find_implication(all_triple_df, triples, observed_triples)
    inverse_patterns, count_inv = find_inverse(all_triple_df, triples, observed_triples)
    # transitive_patterns, count_tran = find_transitive(all_triple_df, triples, observed_triples)

    pd.DataFrame(symmetric_patterns).to_csv(os.path.join(save_pattern_path, 'symmetric_patterns.csv'), sep='\t', header=None, index=False)
    pd.DataFrame(reflexive_patterns).to_csv(os.path.join(save_pattern_path, 'reflexive_patterns.csv'), sep='\t',
                                            header=None, index=False)
    pd.DataFrame(implication_patterns).to_csv(os.path.join(save_pattern_path, 'implication_patterns.csv'), sep='\t',
                                            header=None, index=False)
    pd.DataFrame(inverse_patterns).to_csv(os.path.join(save_pattern_path, 'inverse_patterns.csv'), sep='\t',
                                            header=None, index=False)
    pd.DataFrame(transitive_patterns).to_csv(os.path.join(save_pattern_path, 'transitive_patterns.csv'), sep='\t',
                                            header=None, index=False)




