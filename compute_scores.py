import math

from counting import count_conditionally_independent_triples, count_d_separated_triples
from utils import _get_all_tested_triples


def compute_faithfulness_coherency_score(pc, weight_function=lambda triple: 1):
    all_triples = _get_all_tested_triples(pc.graph.action_history)

    weighted_count_all = sum(weight_function(triple) for triple in all_triples)

    if weighted_count_all == 0:
        return 1  # Avoid division by zero

    weighted_count_ind, ind_triples = count_conditionally_independent_triples(pc, all_triples, weight_function)
    weighted_count_ind_d_sep, _ = count_d_separated_triples(pc, ind_triples, weight_function)

    weighted_count_ind_d_con = weighted_count_ind - weighted_count_ind_d_sep

    return (weighted_count_all - weighted_count_ind_d_con) / weighted_count_all


def compute_markov_coherency_score(pc, weight_function=lambda triple: 1):
    all_triples = _get_all_tested_triples(pc.graph.action_history)

    weighted_count_all = sum(weight_function(triple) for triple in all_triples)

    if weighted_count_all == 0:
        return 1  # Avoid division by zero

    weighted_count_d_sep, d_sep_triples = count_d_separated_triples(pc, all_triples, weight_function)
    weighted_count_ind_d_sep, _ = count_conditionally_independent_triples(pc, d_sep_triples, weight_function)

    weighted_count_d_sep_dep = weighted_count_d_sep - weighted_count_ind_d_sep

    return (weighted_count_all - weighted_count_d_sep_dep) / weighted_count_all


def compute_total_coherency_score(pc, weight_function=lambda triple: 1):
    all_triples = _get_all_tested_triples(pc.graph.action_history)
    for triple in all_triples:
        print(f"all_triple={triple[0].name, triple[1].name, [node.name for node in triple[2]]}, weight={weight_function(triple)}")

    weighted_count_all = sum(weight_function(triple) for triple in all_triples)
    if weighted_count_all == 0:
        return "no tests were performed"

    _, ind_triples = count_conditionally_independent_triples(pc, all_triples)
    for triple in ind_triples:
        print(f"ind_triple={triple[0].name, triple[1].name, [node.name for node in triple[2]]}, weight={weight_function(triple)}")

    _, d_sep_triples = count_d_separated_triples(pc, all_triples)
    for triple in d_sep_triples:
        print(f"d_sep_triple={triple[0].name, triple[1].name, [node.name for node in triple[2]]}, weight={weight_function(triple)}")

    weighted_mismatch = 0
    for triple in ind_triples:
        if triple not in d_sep_triples:
            weighted_mismatch += weight_function(triple)
            print(f"ind_but_d_con_triple={triple[0].name, triple[1].name, [node.name for node in triple[2]]}, weight={weight_function(triple)}")

    for triple in d_sep_triples:
        if triple not in ind_triples:
            weighted_mismatch += weight_function(triple)
            print(
                f"dep_but_d_sep_triple={triple[0].name, triple[1].name, [node.name for node in triple[2]]}, weight={weight_function(triple)}")

    return (weighted_count_all - weighted_mismatch) / weighted_count_all


def weight_by_exponential_decay_of_cardinality_of_conditioning_set(triple):
    return math.exp(-len(triple[2]))

