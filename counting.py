def count_conditionally_independent_triples(pc, subset_of_triples=None, weight_function=lambda triple: 1):
    weighted_count_of_triples = 0
    cond_indep_triples = []
    results = pc.graph.action_history

    for action_history_step in results:
        for action in action_history_step.all_proposed_actions:
            if "separatedBy" in action.data:
                triple = action.data["triple"]
                if subset_of_triples is not None and triple not in subset_of_triples:
                    continue

                cond_indep_triples.append(triple)
                weighted_count_of_triples += weight_function(triple)

    return weighted_count_of_triples, cond_indep_triples


def count_d_separated_triples(pc, subset_of_triples, weight_function=lambda triple: 1):
    weighted_count_of_triples = 0
    d_separated_triples = []

    for triple in subset_of_triples:
        if pc.graph.are_nodes_d_separated_cpdag(triple[0], triple[1], triple[2]):
            d_separated_triples.append(triple)
            weighted_count_of_triples += weight_function(triple)

    return weighted_count_of_triples, d_separated_triples