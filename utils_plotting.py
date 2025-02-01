import matplotlib.pyplot as plt
import numpy as np
from causy.causal_discovery.constraint.algorithms.pc import PCClassic
from compute_rates import compute_total_coherency_rate, compute_faithfulness_coherency_rate, \
    compute_markov_coherency_rate

plt.style.use('seaborn-v0_8-colorblind')
def generate_average_total_coherency_scores(sample_sizes, num_repetitions, model, hidden_variables=None):
    average_coherency_rates = []
    standard_deviations = []
    for sample_size in sample_sizes:
        total_coherency_rates = []
        for _ in range(num_repetitions):

            test_data, graph = model.generate(sample_size)
            if hidden_variables is not None:
                for hidden_variable in hidden_variables:
                    test_data.pop(hidden_variable)

            tst = PCClassic()
            tst.create_graph_from_data(test_data)
            tst.create_all_possible_edges()
            tst.execute_pipeline_steps()

            total_coherency_score = compute_total_coherency_rate(tst)
            total_coherency_rates.append(total_coherency_score)

        average_coherency_rate = np.mean(total_coherency_rates)
        average_coherency_rates.append(average_coherency_rate)

        standard_deviation = np.std(total_coherency_rates)
        standard_deviations.append(standard_deviation)

    return average_coherency_rates, standard_deviations

def generate_average_faithfulness_markov_and_total_coherency_scores(sample_sizes, num_repetitions, model, hidden_variables=None):
    average_faithfulness_coherency_rates = []
    average_markov_coherency_rates = []
    average_total_coherency_rates = []
    for sample_size in sample_sizes:
        faithfulness_coherency_rates = []
        markov_coherency_rates = []
        total_coherency_rates = []
        for _ in range(num_repetitions):

            test_data, graph = model.generate(sample_size)
            if hidden_variables is not None:
                for hidden_variable in hidden_variables:
                    test_data.pop(hidden_variable)

            tst = PCClassic()
            tst.create_graph_from_data(test_data)
            tst.create_all_possible_edges()
            tst.execute_pipeline_steps()

            faithfulness_coherency_score = compute_faithfulness_coherency_rate(tst)
            faithfulness_coherency_rates.append(faithfulness_coherency_score)

            markov_coherency_score = compute_markov_coherency_rate(tst)
            markov_coherency_rates.append(markov_coherency_score)

            total_coherency_score = compute_total_coherency_rate(tst)
            total_coherency_rates.append(total_coherency_score)

        average_faithfulness_coherency_rate = np.mean(faithfulness_coherency_rates)
        average_faithfulness_coherency_rates.append(average_faithfulness_coherency_rate)

        average_markov_coherency_rate = np.mean(markov_coherency_rates)
        average_markov_coherency_rates.append(average_markov_coherency_rate)

        average_total_coherency_rate = np.mean(total_coherency_rates)
        average_total_coherency_rates.append(average_total_coherency_rate)

    return average_faithfulness_coherency_rates, average_markov_coherency_rates, average_total_coherency_rates

def plot_total_coherency_scores_for_different_models(sample_sizes_list, num_repetitions, models, model_names, title, y_lim_from=0, y_lim_to=1.5, hidden_variables=None):
    plt.figure(figsize=(8, 5))

    for model, model_name in zip(models, model_names):
        x_vals = sample_sizes_list
        y_vals, standard_deviations = generate_average_total_coherency_scores(sample_sizes_list, num_repetitions, model, hidden_variables)

        plt.scatter(x_vals, y_vals, s=30, marker='s', label=model_name)  # Smaller square markers
        # plt.errorbar(x_vals, y_vals, yerr=standard_deviations, linestyle='None')

    plt.xlabel("Sample Size")
    plt.ylabel("Average Coherency Score")
    plt.ylim(y_lim_from, y_lim_to)
    plt.title(title)
    plt.legend()  # Add legend
    plt.savefig("plots/" + title + ".png")

def plot_faithfulness_markov_and_total_coherency_scores_for_one_model(sample_sizes_list, num_repetitions, model, title, y_lim_from=0, y_lim_to=1.5, hidden_variables=None):
    plt.figure(figsize=(8, 5))

    x_vals = sample_sizes_list
    y_vals_faithfulness, y_vals_markov, y_vals_total = generate_average_faithfulness_markov_and_total_coherency_scores(sample_sizes_list, num_repetitions, model, hidden_variables)

    plt.scatter(x_vals, y_vals_total, s=30, marker='s', label="Total")  # Smaller square markers
    plt.scatter(x_vals, y_vals_markov, s=30, marker='o', label="Markov")  # Smaller circle markers
    plt.scatter(x_vals, y_vals_faithfulness, s=30, marker='^', label="Faithfulness") # Smaller triangle markers

    plt.xlabel("Sample Size")
    plt.ylabel("Average Coherency Score")
    plt.ylim(y_lim_from, y_lim_to)
    plt.title(title)
    plt.legend()  # Add legend
    # save plot to plots folder
    plt.savefig("plots/" + title + ".png")