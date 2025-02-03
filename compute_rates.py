import itertools

from counting import count_conditionally_independent_triples, count_d_separated_triples

def compute_faithfulness_coherency_rate(pc):
    all_triples = _get_all_tested_triples(pc.graph.action_history)
    nb_of_all_triples = len(all_triples)
    nb_of_ind_triples, ind_triples = count_conditionally_independent_triples(pc, all_triples)
    nb_indep_and_d_sep_triples, _ = count_d_separated_triples(pc, ind_triples)
    nb_of_ind_and_d_con_triples = nb_of_ind_triples - nb_indep_and_d_sep_triples
    if nb_of_ind_triples == 0:
        return 1
    return (nb_of_all_triples - nb_of_ind_and_d_con_triples) / nb_of_all_triples


def compute_markov_coherency_rate(pc):
    all_triples = _get_all_tested_triples(pc.graph.action_history)
    nb_of_all_triples = len(all_triples)
    nb_of_d_sep_triples, d_sep_triples = count_d_separated_triples(pc, all_triples)
    nb_of_ind_and_d_sep_triples, _ = count_conditionally_independent_triples(pc, d_sep_triples)
    nb_of_d_sep_and_dep_triples = nb_of_d_sep_triples - nb_of_ind_and_d_sep_triples
    if nb_of_d_sep_triples == 0:
        return 1
    return (nb_of_all_triples - nb_of_d_sep_and_dep_triples) / nb_of_all_triples


def compute_total_coherency_rate(pc):
    all_tested_triples = _get_all_tested_triples(pc.graph.action_history)
    number_of_tested_triples = len(all_tested_triples)
    if number_of_tested_triples == 0:
        return "no tests were performed"

    nb_ind_triples, ind_triples = count_conditionally_independent_triples(pc)
    nb_d_sep_triples, d_sep_triples = count_d_separated_triples(pc, all_tested_triples)

    counter = 0
    for triple in ind_triples:
        if triple not in d_sep_triples:
            counter += 1

    for triple in d_sep_triples:
        if triple not in ind_triples:
            counter += 1

    return (number_of_tested_triples - counter) / number_of_tested_triples

def _get_all_tested_triples(pc_results):
    triples = []
    for result in pc_results:
        for action in result.all_proposed_actions:
            if "triple" in action.data:
                triples.append(action.data["triple"])

    # the PC algorithm tests ordered triples and their neighbours, therefore some triples are tested twice. We remove these duplicates
    for triple1, triple2 in itertools.combinations(triples, 2):
        if triple1[2] == triple2[2] and triple1[0] == triple2[1] and triple1[1] == triple2[0]:
            triples.remove(triple1)

    return triples

