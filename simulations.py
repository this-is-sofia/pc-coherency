import os

from compute_scores import weight_by_exponential_decay_of_cardinality_of_conditioning_set, \
    compute_total_coherency_score, compute_faithfulness_coherency_score, compute_markov_coherency_score
from generate_average_scores import generate_average_coherency_scores
from models import classical_five_node_example_small_effect_size, \
    classical_five_node_example_medium_effect_size, classical_five_node_example_large_effect_size, \
    faithfulness_violation_four_nodes, faithfulness_violation_five_nodes, faithfulness_violation_six_nodes, \
    causal_insufficiency_three_nodes, causal_insufficiency_four_nodes, faithfulness_violation_three_nodes, \
    mediated_path_very_small_effects, mediated_path_small_effects, mediated_path_almost_one_as_effect, \
    mediated_path_one_as_effect, mediated_path_five_as_effect, mediated_path_ten_as_effect, \
    classical_five_node_example_one_fifth_effect_size, classical_five_node_example_one_half_effect_size
from plotting import plot_coherency_scores

sample_sizes_list = [50, 100, 1000, 10000]
num_repetitions = 100
score_functions = {
    "Total": compute_total_coherency_score,
    "Faithfulness": compute_faithfulness_coherency_score,
    "Markov": compute_markov_coherency_score
}

if __name__ == "__main__":

    # Simulations for mediated effects
    model_names_mediated_effects = [
        "effect_size_0.2",
        "effect_size_0.5",
        "effect_size_1",
        "effect_size_5",
        "effect_size_10"
    ]
    effect_models_mediated_path = [
        mediated_path_very_small_effects,
        mediated_path_small_effects,
        mediated_path_one_as_effect,
        mediated_path_five_as_effect,
        mediated_path_ten_as_effect
        ]

    # Define hidden variable sets for different chain lengths
    hidden_variable_sets = [
        ["W", "V", "U", "T"],    # Three nodes (X → Y → Z)
        ["V", "U", "T"],         # Four nodes (X → Y → Z → W)
        ["U", "T"],              # Five nodes (X → Y → Z → W → V)
        ["T"],                   # Six nodes (X → Y → Z → W → V → U)
        []                       # Seven nodes (X → Y → Z → W → V → U → T)
    ]

    # Generate plots for different node configurations

    for hidden_vars in hidden_variable_sets:
        for model, model_name_string in zip(effect_models_mediated_path, model_names_mediated_effects):
            generate_average_coherency_scores(
                sample_sizes_list,
                num_repetitions,
                model,
                score_functions,
                hidden_variables=hidden_vars,
                model_name=model_name_string,
                folder_name="mediated_path"
            )

    # Simulations for classical five node model
    effect_models_five_nodes = [
        classical_five_node_example_one_fifth_effect_size,
        classical_five_node_example_one_half_effect_size,
        classical_five_node_example_small_effect_size,
        classical_five_node_example_medium_effect_size,
        classical_five_node_example_large_effect_size
    ]

    model_names_five_nodes = [
        "effect size 0.2",
        "effect size 0.5",
        "effect size 1",
        "effect size 5",
        "effect size 10"
    ]

    for model, model_name_string in zip(effect_models_five_nodes, model_names_five_nodes):
        generate_average_coherency_scores(
            sample_sizes_list,
            num_repetitions,
            model,
            score_functions,
            hidden_variables=[],
            model_name=model_name_string,
            folder_name="five_node_model"
        )

    # Simulations for faithfulness violations
    effect_models_faithfulness_violation = [
        faithfulness_violation_three_nodes,
        faithfulness_violation_four_nodes,
        faithfulness_violation_five_nodes,
        faithfulness_violation_six_nodes
    ]

    model_names_faithfulness_violation = [
        "three nodes",
        "four nodes",
        "five nodes",
        "six nodes"
    ]

    for model, model_name_string in zip(effect_models_faithfulness_violation, model_names_faithfulness_violation):
        generate_average_coherency_scores(
            sample_sizes_list,
            num_repetitions,
            model,
            score_functions,
            hidden_variables=[],
            model_name=model_name_string,
            folder_name="faithfulness_violation"
        )
    # Simulation for causal insufficiency, three hidden variables (U_1, U_2, U_3)
    generate_average_coherency_scores(
        sample_sizes_list,
        num_repetitions,
        causal_insufficiency_three_nodes,
        score_functions,
        hidden_variables=["U1", "U2", "U3"],
        model_name="three nodes",
        folder_name="causal_insufficiency"
    )

    # Simulation for causal insufficiency, four hidden variables (U_1, U_2, U_3, U_4)
    generate_average_coherency_scores(
        sample_sizes_list,
        num_repetitions,
        causal_insufficiency_four_nodes,
        score_functions,
        hidden_variables=["U1", "U2", "U3", "U4"],
        model_name="four nodes",
        folder_name="causal_insufficiency"
    )

    # Plot all results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_results = os.path.join(script_dir, "results")
    plot_coherency_scores(results_folder=folder_results, output_folder="plots")