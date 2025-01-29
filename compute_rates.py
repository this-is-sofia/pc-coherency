from utils_counting import count_conditionally_independent_triples, count_d_separated_triples

def compute_faithfulness_coherency_rate(pc):
    nb_of_ind_triples, triples = count_conditionally_independent_triples(pc)
    nb_indep_and_d_sep_triples, _ = count_d_separated_triples(pc, triples)
    if nb_of_ind_triples == 0:
        return 1
    return nb_indep_and_d_sep_triples / nb_of_ind_triples


def compute_markov_coherency_rate(pc):
    nb_of_d_sep_triples, triples = count_d_separated_triples(pc, _get_all_tested_triples(pc.graph.action_history))
    nb_of_ind_triples, _ = count_conditionally_independent_triples(pc, triples)
    if nb_of_d_sep_triples == 0:
        return 1
    return nb_of_ind_triples / nb_of_d_sep_triples


def compute_total_coherency_rate(pc):
    all_tested_triples = _get_all_tested_triples(pc.graph.action_history)
    number_of_tested_triples = len(all_tested_triples)
    if number_of_tested_triples == 0:
        return "no tests were performed"

    nb_of_tested_d_sep_triples, _ = count_d_separated_triples(pc, all_tested_triples)
    nb_of_ind_triples, _ = count_conditionally_independent_triples(pc)

    return (number_of_tested_triples - abs(nb_of_tested_d_sep_triples - nb_of_ind_triples)) / number_of_tested_triples




def _get_all_tested_triples(pc_results):
    triples = []
    for result in pc_results:
        for action in result.all_proposed_actions:
            if "triple" in action.data:
                triples.append(action.data["triple"])
    return triples

