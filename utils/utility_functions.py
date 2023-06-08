import numpy as np
import pandas as pd


def build_indexed_dictioary(data):
    """
    :param data: the triple data from the dataset
    :return: the indexed dictionary where we mark the existing triples (observed triples) as true or 1
    """
    triples = list(
        zip([h for h in data[0]], [r for r in data[1]], [t for t in data[2]])
    )
    observed_triples = {}
    for triple in triples:
        observed_triples[triple[0], triple[1], triple[2]] = 1
    return triples, observed_triples


def find_reflexive(data):
    """
    :param data: The dataset in the triple format
    :return: the reflexive patterns and the count of the reflexive patterns per relation or None, None if no reflexive pattern is found
    """
    reflexive_triples = data.loc[data[0] == data[2]]
    reflexive_triples = pd.DataFrame(
        np.array(reflexive_triples), columns=["head", "relation", "tail"]
    )
    unique_relations_reflexive_triples = np.array(
        list(set(list(reflexive_triples["relation"])))
    )

    reflexive_triple_per_relation_count = []
    for unique_relations_reflexive_triple in unique_relations_reflexive_triples:
        # print(unique_relations_reflexive_triple)
        place_holder_relation_triple = reflexive_triples.loc[
            reflexive_triples["relation"] == unique_relations_reflexive_triple
        ]
        place_holder_relation_triple = np.array(place_holder_relation_triple)
        # print(unique_relations_reflexive_triple, ' : ', len(place_holder_relation_triple))
        stat_count_per_relation = np.array(
            [unique_relations_reflexive_triple, int(len(place_holder_relation_triple))]
        )
        reflexive_triple_per_relation_count.append(stat_count_per_relation)
    if len(reflexive_triples) == 0:
        return None, None
    reflexive_triple_per_relation = pd.DataFrame(
        np.array(reflexive_triple_per_relation_count)
    )
    # reflexive_triple_per_relation['reflexive_relation_count'] = reflexive_triple_per_relation['reflexive_relation_count'].astype(str).astype(int)
    # reflexive_triples_per_relation_sorted = reflexive_triple_per_relation.sort_values(by='reflexive_relation_count', ascending=False)

    return reflexive_triples, reflexive_triple_per_relation


def find_implication(triples_df, triples, observed_triples):
    """
    :param triples_df: the dataset in triple for taking all relation into account
    :param triples: the dataset in triple format where we will find the implication patterns
    :param observed_triples: existing triples marked as true or 1
    :return: the implication patterns and their count per relation or None, None if no implication pattern is found
    """
    all_relations = list(set(list(triples_df[1])))
    premise_list = []
    conclusion_list = []
    implication_patterns = []
    # count = 0
    # for implication
    for triple in triples:
        # print(count)
        for r2 in all_relations:
            if r2 != triple[1]:
                try:
                    exists = observed_triples[triple[0], r2, triple[2]]
                    premise = np.array([triple[0], triple[1], triple[2]])
                    conclusion = np.array([triple[0], r2, triple[2]])
                    # print('premise ', premise)
                    # print('conclusion ', conclusion)
                    # print('######################################')
                    premise_list.append(np.array(premise))
                    conclusion_list.append(np.array(conclusion))
                    implication_pattern = [
                        triple[0],
                        triple[1],
                        triple[2],
                        "->",
                        triple[0],
                        r2,
                        triple[2],
                    ]
                    implication_patterns.append(implication_pattern)
                except:
                    continue
        # count += 1
    if len(implication_patterns) == 0:
        return None, None
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
            [premise, int(len(implication_array_place_holder))]
        )
        implication_triple_per_relation.append(stat_count_per_relation_implication)

    implication_triple_per_relation = pd.DataFrame(
        np.array(implication_triple_per_relation)
    )
    # implication_triple_per_relation[1] = implication_triple_per_relation[1].astype(str).astype(int)
    # implication_triple_per_relation_sorted = implication_triple_per_relation.sort_values(by=1, ascending=False)

    return implication_patterns, implication_triple_per_relation


def find_inverse(triples_df, triples, observed_triples):
    """
    :param triples_df: the dataset in triple for taking all relation into account
    :param triples: the dataset in triple format where we will find the inverse patterns
    :param observed_triples: existing triples marked as true or 1
    :return: the inverse patterns and their count per relation or None, None if no inverse patterns are found
    """
    # triples_df = pd.DataFrame(static)
    all_relations = list(set(list(triples_df[1])))
    premise_list = []
    conclusion_list = []
    inverse_patterns = []
    inverse_relations = []

    for triple in triples:
        for r2 in all_relations:
            if r2 != triple[1]:
                try:
                    exists = observed_triples[triple[2], r2, triple[0]]
                    premise = np.array([triple[0], triple[1], triple[2]])
                    conclusion = np.array([triple[2], r2, triple[0]])
                    # print('premise ', premise)
                    # print('conclusion ', conclusion)
                    # print('######################################')
                    premise_list.append(np.array(premise))
                    conclusion_list.append(np.array(conclusion))
                    inverse_pattern = [
                        triple[0],
                        triple[1],
                        triple[2],
                        "->",
                        triple[2],
                        r2,
                        triple[0],
                    ]
                    inverse_patterns.append(inverse_pattern)
                except:
                    continue
    if len(inverse_patterns) == 0:
        return None, None
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
            [premise, int(len(inverse_array_place_holder))]
        )
        inverse_triple_per_relation.append(stat_count_per_relation_inverse)

    inverse_triple_per_relation = pd.DataFrame(np.array(inverse_triple_per_relation))
    inverse_triple_per_relation[1] = (
        inverse_triple_per_relation[1].astype(str).astype(int)
    )

    return inverse_patterns, inverse_triple_per_relation


def find_symmetric(triples, observed_triples):
    """
    :param triples_df: the dataset in triple for taking all relation into account
    :param triples: the dataset in triple format where we will find the symmetric patterns
    :param observed_triples: existing triples marked as true or 1
    :return: the symmmetric patterns and their count per relation or None, None if no symmetric patterns are found
    """
    triples_df = pd.DataFrame(triples)
    premise_list = []
    conclusion_list = []
    symmetric_patterns = []
    symmetric_relations = []
    for triple in triples:
        # print(triple)
        try:
            exists = observed_triples[triple[2], triple[1], triple[0]]
            # print('original_triple ', triple)
            # symmetric = np.array([triple[2], triple[1], triple[0]])
            premise = np.array([triple[0], triple[1], triple[2]])
            conclusion = np.array([triple[2], triple[1], triple[0]])
            # print('premise ', premise)
            # print('conclusion ', conclusion)
            symmtetric_pattern = [
                triple[0],
                triple[1],
                triple[2],
                "->",
                triple[2],
                triple[1],
                triple[0],
            ]
            symmetric_patterns.append(symmtetric_pattern)
            symmetric_relations.append(triple[1])
            # print('######################################')
            premise_list.append(np.array(premise))
            conclusion_list.append(np.array(conclusion))
        except:
            continue
    if len(symmetric_patterns) == 0:
        return None, None
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
            [premise, int(len(symmetric_array_place_holder))]
        )
        symmetric_triple_per_relation.append(stat_count_per_relation_symmetric)

    symmetric_triple_per_relation = pd.DataFrame(
        np.array(symmetric_triple_per_relation)
    )
    symmetric_triple_per_relation[1] = (
        symmetric_triple_per_relation[1].astype(str).astype(int)
    )
    symmetric_triple_per_relation_sorted = symmetric_triple_per_relation.sort_values(
        by=1, ascending=False
    )

    return symmetric_patterns, symmetric_triple_per_relation_sorted


def find_composition(triples_df, triples, observed_triples):
    """
    :param triples_df: the dataset in triple for taking all relation into account
    :param triples: the dataset in triple format where we will find the composition patterns
    :param observed_triples: existing triples marked as true or 1
    :return: the composition patterns and their count per relation or None, None if no composition patterns
    """
    # print(triples_df)
    # all_relations = list(set(list(triples_df[1])))
    triples_df.columns = ["h", "r", "t"]
    premise_list = []
    conclusion_list = []
    composition_patterns = []
    composition_relations = []
    merged_df = pd.merge(
        triples_df, triples_df, left_on=["t"], right_on=["h"], how="inner"
    )
    merged_data = merged_df.to_numpy()
    for triple in merged_data:
        try:
            exists = observed_triples[triple[0], triple[4], triple[5]]
            premise = np.array(
                [triple[0], triple[1], triple[2], triple[3], triple[4], triple[5]]
            )
            conclusion = np.array([triple[0], triple[4], triple[5]])
            composition_pattern = [
                triple[0],
                triple[1],
                triple[2],
                "and",
                triple[3],
                triple[4],
                triple[5],
                "->",
                triple[0],
                triple[4],
                triple[5],
            ]
            composition_patterns.append(composition_pattern)
            composition_relations.append([triple[1], triple[4]])
            premise_list.append(np.array(premise))
            conclusion_list.append(np.array(conclusion))
        except:
            continue
    if len(composition_patterns) == 0:
        return None, None
    premise_df = pd.DataFrame(np.array(premise_list)).applymap(str)
    conclusion_df = pd.DataFrame(np.array(conclusion_list)).applymap(str)
    premise_df_unique_relation = np.array(list(set([str(x) for x in premise_df[1]])))
    conclusion_df_unique_relation = np.array(
        list(set([str(x) for x in conclusion_df[1]]))
    )

    #####################################################################################################
    composition_triple_per_relation = []
    for premise in premise_df_unique_relation:
        composition_array_place_holder = premise_df.loc[premise_df[1] == premise]
        composition_array_place_holder = np.array(composition_array_place_holder)
        stat_count_per_relation_composition = np.array(
            [premise, int(len(composition_array_place_holder))]
        )
        composition_triple_per_relation.append(stat_count_per_relation_composition)

    composition_triple_per_relation = pd.DataFrame(
        np.array(composition_triple_per_relation)
    )
    composition_triple_per_relation[1] = (
        composition_triple_per_relation[1].astype(str).astype(int)
    )
    composition_triple_per_relation_sorted = (
        composition_triple_per_relation.sort_values(by=1, ascending=False)
    )

    return composition_patterns, composition_triple_per_relation_sorted


def find_transitive(triples_df, triples, observed_triples):
    """
    :param triples_df: the dataset in triple for taking all relation into account
    :param triples: the dataset in triple format where we will find the transitive patterns
    :param observed_triples: existing triples marked as true or 1
    :return: the transitive patterns and their count per relation or None, None if there is no transitive pattern
    """
    all_relations = list(set(list(triples_df[1])))
    triples_df.columns = ["h", "r", "t"]
    premise_list = []
    conclusion_list = []
    transitive_patterns = []
    transitive_relations = []
    merged_df = pd.merge(
        triples_df, triples_df, left_on=["r", "t"], right_on=["r", "h"], how="inner"
    )

    # print('triples_df', triples_df)
    # print('##########################')
    # print('merged_df_trans', merged_df)
    merged_data = merged_df.to_numpy()
    for triple in merged_data:
        try:
            exists = observed_triples[triple[0], triple[1], triple[4]]
            premise = np.array(
                [triple[0], triple[1], triple[2], triple[3], triple[1], triple[4]]
            )
            conclusion = np.array([triple[0], triple[1], triple[4]])
            transitive_pattern = [
                triple[0],
                triple[1],
                triple[2],
                "and",
                triple[3],
                triple[1],
                triple[4],
                "->",
                triple[0],
                triple[1],
                triple[4],
            ]
            transitive_patterns.append(transitive_pattern)
            transitive_relations.append(triple[1])
            premise_list.append(np.array(premise))
            conclusion_list.append(np.array(conclusion))
        except:
            continue
    if len(transitive_patterns) == 0:
        return None, None
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
            [premise, int(len(transitive_array_place_holder))]
        )
        transitive_triple_per_relation.append(stat_count_per_relation_transitive)

    transitive_triple_per_relation = pd.DataFrame(
        np.array(transitive_triple_per_relation)
    )
    transitive_triple_per_relation[1] = (
        transitive_triple_per_relation[1].astype(str).astype(int)
    )
    transitive_triple_per_relation_sorted = transitive_triple_per_relation.sort_values(
        by=1, ascending=False
    )

    return transitive_patterns, transitive_triple_per_relation_sorted
