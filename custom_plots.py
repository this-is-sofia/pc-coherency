import json
import os

from matplotlib import pyplot as plt

from compute_scores import compute_total_coherency_score, compute_faithfulness_coherency_score, \
    compute_markov_coherency_score
from generate_average_scores import generate_average_coherency_scores
from models import classical_five_node_example_small_effect_size, mediated_path_one_as_effect

sample_sizes_list = [10, 31, 100, 316, 1000, 3160, 10000]
num_repetitions = 100
score_functions = {
    "Total": compute_total_coherency_score,
    "Faithfulness": compute_faithfulness_coherency_score,
    "Markov": compute_markov_coherency_score
}

if __name__ == "__main__":

    """
    generate_average_coherency_scores(
        sample_sizes_list,
        num_repetitions,
        classical_five_node_example_small_effect_size,
        score_functions,
        hidden_variables=[],
        model_name="five_node_example_for_plot",
        folder_name="results_for_custom_plots"
    )
    """
    file_path = "results/results_for_custom_plots/five_node_example_for_plot_no_hidden_vars.json"

    # Load the json file
    with open(file_path, "r") as f:
        data = json.load(f)

    sample_sizes = data["sample_sizes"]

    plt.figure(figsize=(6, 5))

    for key in data["scores"]:
        if key == "Orientation Conflicts":
            continue
        # plot data but with logarithmic scale on x-axis
        plt.scatter(sample_sizes, data["scores"][key][0], s=20, marker="s", label=key)

    plt.xscale("log")
    plt.xlabel("Sample Size", fontsize=16)
    plt.ylabel("Average Coherency Rate", fontsize=16)
    plt.ylim(0.6, 1.05)
    plt.title("Coherency Rates for Toy Model", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)  # Add a subtle grid

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save the plot
    if not os.path.exists("custom_plots"):
        os.makedirs("custom_plots")
    plot_filename = "five_nodes.png"
    plt.savefig(os.path.join("custom_plots", plot_filename), dpi=600, bbox_inches="tight")
    plt.close()

    """
    generate_average_coherency_scores(
        sample_sizes_list,
        num_repetitions,
        mediated_path_one_as_effect,
        score_functions,
        hidden_variables=["W", "V", "U", "T"],
        model_name="mediated_effect_example_for_plot",
        folder_name="results_for_custom_plots"
    )
    
    file_path = "results/results_for_custom_plots/mediated_effect_example_for_plot_W_V_U_T.json"

    # Load the json file
    with open(file_path, "r") as f:
        data = json.load(f)

    sample_sizes = data["sample_sizes"]

    plt.figure(figsize=(6, 4.5))

    for key in data["scores"]:
        if key == "Orientation Conflicts":
            continue
        # plot data but with logarithmic scale on x-axis

        plt.scatter(sample_sizes, data["scores"][key][0], s=5, marker="s", label=key)

    plt.xscale("log")
    plt.xlabel("Sample Size")
    plt.ylabel("Average Coherency Rate")
    plt.ylim(0.6, 1.05)
    plt.title("Coherency Rates for Mediated Effect")
    plt.legend()

    # Save the plot
    plot_filename = "mediated_effect.png"
    plt.savefig(os.path.join("custom_plots", "mediated_effect.png"))
    plt.close()
    """