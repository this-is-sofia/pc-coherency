from models import classical_five_node_example_small_effect_size, \
    classical_five_node_example_medium_effect_size, classical_five_node_example_large_effect_size, \
    faithfulness_violation_four_nodes, faithfulness_violation_five_nodes, faithfulness_violation_six_nodes, \
    causal_insufficiency_three_nodes, causal_insufficiency_four_nodes, faithfulness_violation_three_nodes, \
    mediated_path_very_small_effects, mediated_path_small_effects, mediated_path_almost_one_as_effect, \
    mediated_path_one_as_effect, mediated_path_five_as_effect, mediated_path_ten_as_effect
from plotting import plot_total_coherency_scores_for_different_models, plot_coherency_scores

sample_sizes_list = [i for i in range(100, 1600, 100)]
num_repetitions = 50
model_names = ["effect size 0.2", "effect size 0.5", "effect size 1", "effect size 5", "effect size 10"]
effect_models = [
    mediated_path_very_small_effects,
    mediated_path_small_effects,
    mediated_path_almost_one_as_effect,
    mediated_path_one_as_effect,
    mediated_path_five_as_effect,
    mediated_path_ten_as_effect
    ]

# Define hidden variable sets for different chain lengths
hidden_variable_sets = [
    ["W", "V", "U", "T"],    # Three nodes (X → Y → Z)
    ["V", "U", "T"],         # Four nodes (X → Y → Z → W)
    #["U", "T"],              # Five nodes (X → Y → Z → W → V)
    #["T"],                   # Six nodes (X → Y → Z → W → V → U)
    #[]                       # Seven nodes (X → Y → Z → W → V → U → T)
]

# Generate plots for different node configurations
for i, hidden_vars in enumerate(hidden_variable_sets, start=3):
    title = f"Total Coherency Scores for Mediated Effect With {i} Nodes"
    plot_total_coherency_scores_for_different_models(
        sample_sizes_list,
        num_repetitions,
        effect_models,
        model_names,
        title,
        hidden_variables=hidden_vars,
        y_lim_from=0.5,
        y_lim_to=1.2
    )


# Example five nodes and X -> Z <-Y as well as V <- Z -> W, different effect sizes, compare total scores
sample_sizes_list = [i for i in range(1000, 56000, 5000)]
num_repetitions = 50
model_names = ["small effects", "medium effects", "large effects"]
#plot_total_coherency_scores_for_different_models(sample_sizes_list, num_repetitions, [classical_five_node_example_small_effect_size, classical_five_node_example_medium_effect_size, classical_five_node_example_large_effect_size], model_names, "Total Coherency Scores for Five Nodes With Different Effect Sizes")

# Example for faithfulness violation with three nodes X -> Y -> Z, X-> Z, one model, all three scores
sample_sizes_list = [i for i in range(100, 3400, 400)]
num_repetitions = 50
#plot_coherency_scores(sample_sizes_list, num_repetitions, faithfulness_violation_three_nodes, "Coherency Scores for Faithfulness Violation (three nodes)", y_lim_from=0.6, y_lim_to=1.1)

# Example for faithfulness violation with four nodes X -> V -> W -> Y, X-> Y, one model, all three scores
sample_sizes_list = [i for i in range(100, 3400, 400)]
num_repetitions = 50
#plot_coherency_scores(sample_sizes_list, num_repetitions, faithfulness_violation_four_nodes, "Coherency Scores for Faithfulness Violation (four nodes)", y_lim_from=0.6, y_lim_to=1.1)

# Example for faithfulness violation with five nodes X -> V -> W -> U -> Y, X-> Y, one model, all three scores
sample_sizes_list = [i for i in range(100, 3400, 400)]
num_repetitions = 50
#plot_coherency_scores(sample_sizes_list, num_repetitions, faithfulness_violation_five_nodes, "Coherency Scores for Faithfulness Violation (five nodes)", y_lim_from=0.6, y_lim_to=1.1)

# Example for faithfulness violation with six nodes X -> V -> W -> U -> Z -> Y, X-> Y, one model, all three scores
sample_sizes_list = [i for i in range(100, 3400, 400)]
num_repetitions = 50
#plot_coherency_scores(sample_sizes_list, num_repetitions, faithfulness_violation_six_nodes, "Coherency Scores for Faithfulness Violation (six nodes)", y_lim_from=0.6, y_lim_to=1.1)

# Example for causal insufficiency with three nodes which are confounded by a hidden variable pairwise, one model, all three scores
sample_sizes_list = [i for i in range(100, 2100, 100)]
num_repetitions = 50
#plot_coherency_scores(sample_sizes_list, num_repetitions, causal_insufficiency_three_nodes, "Coherency Scores for Causal Insufficiency (three nodes)", y_lim_from=0.8, y_lim_to=1.1, hidden_variables=["U1", "U2", "U3"])

# Example for causal insufficiency with four nodes which are confounded by a hidden variable pairwise, one model, all three scores
sample_sizes_list = [i for i in range(100, 2100, 100)]
num_repetitions = 50
#plot_coherency_scores(sample_sizes_list, num_repetitions, causal_insufficiency_four_nodes, "Coherency Scores for Causal Insufficiency (four nodes)", y_lim_from=0.8, y_lim_to=1.1, hidden_variables=["U1", "U2", "U3", "U4"])
