import json
import os

from causy.causal_discovery.constraint.algorithms.pc import PCClassic, PC, PCStable
from causy.edge_types import DirectedEdge

from compute_scores import compute_total_coherency_score, compute_markov_coherency_score, \
    compute_faithfulness_coherency_score, weight_by_exponential_decay_of_cardinality_of_conditioning_set
from utils import _get_number_of_orientation_conflicts


def make_score_dictionary(real_world_data_set):

    # run pc on real world data set
    pc = PC()
    pc.create_graph_from_data(real_world_data_set)
    pc.create_all_possible_edges()
    pc.execute_pipeline_steps()

    # make dictionary of scores
    scores = {}
    scores["Total Coherency Rate"] = compute_total_coherency_score(pc)
    scores["Markov Coherency Rate"] = compute_markov_coherency_score(pc)
    scores["Faithfulness Coherency Rate"] = compute_faithfulness_coherency_score(pc)
    scores["Weighted Total Coherency Score"] = compute_total_coherency_score(pc, weight_by_exponential_decay_of_cardinality_of_conditioning_set)
    scores["Weighted Markov Coherency Score"] = compute_markov_coherency_score(pc, weight_by_exponential_decay_of_cardinality_of_conditioning_set)
    scores["Weighted Faithfulness Coherency Score"] = compute_faithfulness_coherency_score(pc, weight_by_exponential_decay_of_cardinality_of_conditioning_set)
    scores["Number of Orientation Conflicts"] = _get_number_of_orientation_conflicts(pc.graph.action_history)

    return scores


if __name__ == "__main__":
    # Load auto_mpg data set
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_auto_mpg = os.path.join(script_dir, "auto_mpg")
    file_path_auto_mpg = os.path.join(folder_auto_mpg, "auto_mpg.json")

    # Load the JSON file
    with open(file_path_auto_mpg, "r", encoding="utf-8") as f:
        auto_mpg_data_set = json.load(f)

    # Load the hotel data sets
    folder_hotel = os.path.join(script_dir, "hotels")
    file_path_hotels_1 = os.path.join(folder_hotel, "h1.json")
    file_path_hotels_2 = os.path.join(folder_hotel, "h2.json")

    # Load the JSON files
    with open(file_path_hotels_1, "r", encoding="utf-8") as f:
        hotel_data_set_1 = json.load(f)
    with open(file_path_hotels_2, "r", encoding="utf-8") as f:
        hotel_data_set_2 = json.load(f)

    # Make dictionary of scores
    scores_auto_mpg = make_score_dictionary(auto_mpg_data_set)
    #scores_hotels_1 = make_score_dictionary(hotel_data_set_1)
    #scores_hotels_2 = make_score_dictionary(hotel_data_set_2)

    # Print scores
    print(scores_auto_mpg)
    #print(scores_hotels_1)
    #print(scores_hotels_2)

