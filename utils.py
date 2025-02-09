import itertools


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


def _get_number_of_orientation_conflicts(pc_results):
    conflicts = 0
    for result in pc_results:
        for action in result.all_proposed_actions:
            if "orientation_conflict" in action.data:
                conflicts += 1

    return conflicts
