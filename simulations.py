from models import mediated_effect_small_effect_sizes, mediated_effect_medium_effect_sizes, \
    mediated_effect_large_effect_sizes, classical_five_node_example_small_effect_size, \
    classical_five_node_example_medium_effect_size, classical_five_node_example_large_effect_size, \
    faithfulness_violation_four_nodes, faithfulness_violation_five_nodes, faithfulness_violation_six_nodes, \
    causal_insufficiency_three_nodes, causal_insufficiency_four_nodes, faithfulness_violation_three_nodes
from utils_plotting import plot_total_coherency_scores_for_different_models, \
    plot_faithfulness_markov_and_total_coherency_scores_for_one_model

# Example three nodes X -> Y -> Z, different effect sizes, compare total scores
sample_sizes_list = [i for i in range(50, 1050, 50)]
num_repetitions = 50
model_names = ["small effects", "medium effects", "large effects"]
plot_total_coherency_scores_for_different_models(sample_sizes_list, num_repetitions, [mediated_effect_small_effect_sizes, mediated_effect_medium_effect_sizes, mediated_effect_large_effect_sizes], model_names, "Total Coherency Scores for Mediated Effect With Different Effect Sizes")

# Example five nodes and X -> Z <-Y as well as V <- Z -> W, different effect sizes, compare total scores
sample_sizes_list = [i for i in range(1000, 56000, 5000)]
num_repetitions = 50
model_names = ["small effects", "medium effects", "large effects"]
plot_total_coherency_scores_for_different_models(sample_sizes_list, num_repetitions, [classical_five_node_example_small_effect_size, classical_five_node_example_medium_effect_size, classical_five_node_example_large_effect_size], model_names, "Total Coherency Scores for Five Nodes With Different Effect Sizes")

# Example for faithfulness violation with three nodes X -> Y -> Z, X-> Z, one model, all three scores
sample_sizes_list = [i for i in range(100, 3400, 400)]
num_repetitions = 50
plot_faithfulness_markov_and_total_coherency_scores_for_one_model(sample_sizes_list, num_repetitions, faithfulness_violation_three_nodes, "Coherency Scores for Faithfulness Violation (three nodes)", y_lim_from=0.6, y_lim_to=1.1)

# Example for faithfulness violation with four nodes X -> V -> W -> Y, X-> Y, one model, all three scores
sample_sizes_list = [i for i in range(100, 3400, 400)]
num_repetitions = 50
plot_faithfulness_markov_and_total_coherency_scores_for_one_model(sample_sizes_list, num_repetitions, faithfulness_violation_four_nodes, "Coherency Scores for Faithfulness Violation (four nodes)", y_lim_from=0.6, y_lim_to=1.1)

# Example for faithfulness violation with five nodes X -> V -> W -> U -> Y, X-> Y, one model, all three scores
sample_sizes_list = [i for i in range(100, 3400, 400)]
num_repetitions = 50
plot_faithfulness_markov_and_total_coherency_scores_for_one_model(sample_sizes_list, num_repetitions, faithfulness_violation_five_nodes, "Coherency Scores for Faithfulness Violation (five nodes)", y_lim_from=0.6, y_lim_to=1.1)

# Example for faithfulness violation with six nodes X -> V -> W -> U -> Z -> Y, X-> Y, one model, all three scores
sample_sizes_list = [i for i in range(100, 3400, 400)]
num_repetitions = 50
plot_faithfulness_markov_and_total_coherency_scores_for_one_model(sample_sizes_list, num_repetitions, faithfulness_violation_six_nodes, "Coherency Scores for Faithfulness Violation (six nodes)", y_lim_from=0.6, y_lim_to=1.1)

# Example for causal insufficiency with three nodes which are confounded by a hidden variable pairwise, one model, all three scores
sample_sizes_list = [i for i in range(100, 2100, 100)]
num_repetitions = 50
plot_faithfulness_markov_and_total_coherency_scores_for_one_model(sample_sizes_list, num_repetitions, causal_insufficiency_three_nodes, "Coherency Scores for Causal Insufficiency (three nodes)", y_lim_from=0.8, y_lim_to=1.1, hidden_variables=["U1", "U2", "U3"])

# Example for causal insufficiency with four nodes which are confounded by a hidden variable pairwise, one model, all three scores
sample_sizes_list = [i for i in range(100, 2100, 100)]
num_repetitions = 50
plot_faithfulness_markov_and_total_coherency_scores_for_one_model(sample_sizes_list, num_repetitions, causal_insufficiency_four_nodes, "Coherency Scores for Causal Insufficiency (four nodes)", y_lim_from=0.8, y_lim_to=1.1, hidden_variables=["U1", "U2", "U3", "U4"])
