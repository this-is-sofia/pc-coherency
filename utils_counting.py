import itertools


def count_conditionally_independent_triples(pc, subset_of_triples=None):
    nb_of_triples = 0
    cond_indep_triples = []
    results = pc.graph.action_history

    for action_history_step in results:

        for action in action_history_step.actions:

            if subset_of_triples is not None:
                # the conditioning set of a conditionally independent triple is stored in the "separatedBy" key
                if "separatedBy" in action.data:
                    if action.data["triple"] in subset_of_triples:
                        cond_indep_triples.append(action.data["triple"])
                        nb_of_triples += 1

            elif "separatedBy" in action.data:
                cond_indep_triples.append(action.data["triple"])
                nb_of_triples += 1

    return (nb_of_triples, cond_indep_triples)

def count_d_separated_triples(pc, subset_of_triples):
    nb_of_triples = 0
    d_separated_triples = []
    
    # if subset_of_triples is given, check only those triples
    if subset_of_triples is not None:
        for triple in subset_of_triples:
            if pc.graph.are_nodes_d_separated_cpdag(triple[0], triple[1], triple[2]):
                d_separated_triples.append(triple)
                nb_of_triples += 1
    return (nb_of_triples, d_separated_triples)