import matplotlib.pyplot as plt
import numpy as np
from causy.causal_discovery.constraint.algorithms.pc import PCClassic
from compute_rates import compute_total_coherency_rate, compute_faithfulness_coherency_rate, \
    compute_markov_coherency_rate

# Set default plot style
plt.style.use('seaborn-v0_8-colorblind')


def generate_test_data_and_run_pc(model, sample_size, hidden_variables=None):
    """
    Generates test data from the model, removes hidden variables if provided,
    and runs the PCClassic causal discovery algorithm.

    :param model: The data-generating model.
    :param sample_size: Number of samples.
    :param hidden_variables: List of variables to remove from the data.
    :return: Processed PCClassic instance.
    """
    test_data, _ = model.generate(sample_size)

    if hidden_variables:
        for hidden_variable in hidden_variables:
            test_data.pop(hidden_variable, None)

    tst = PCClassic()
    tst.create_graph_from_data(test_data)
    tst.create_all_possible_edges()
    tst.execute_pipeline_steps()

    return tst


def generate_average_coherency_scores(sample_sizes, num_repetitions, model, score_functions, hidden_variables=None):
    """
    Computes the average and standard deviation of coherency scores using given scoring functions.

    :param sample_sizes: List of sample sizes.
    :param num_repetitions: Number of repetitions per sample size.
    :param model: The model to generate data.
    :param score_functions: Dictionary with {score_name: score_function}.
    :param hidden_variables: List of hidden variables to exclude from test data.
    :return: Dictionary {score_name: (averages, standard_deviations)}
    """
    results = {name: ([], []) for name in score_functions.keys()}  # Initialize lists

    for sample_size in sample_sizes:
        scores = {name: [] for name in score_functions.keys()}  # Store scores per iteration

        for _ in range(num_repetitions):
            tst = generate_test_data_and_run_pc(model, sample_size, hidden_variables)

            for name, func in score_functions.items():
                scores[name].append(func(tst))

        for name in score_functions.keys():
            results[name][0].append(np.mean(scores[name]))  # Average
            results[name][1].append(np.std(scores[name]))  # Standard deviation

    return results


def plot_coherency_scores(sample_sizes_list, num_repetitions, model, title, y_lim_from=0, y_lim_to=1.5,
                          hidden_variables=None):
    """
    Plots coherency scores (Total, Faithfulness, Markov) for a given model.

    :param sample_sizes_list: List of sample sizes.
    :param num_repetitions: Number of repetitions per sample size.
    :param model: The model to generate data.
    :param title: Title of the plot.
    :param y_lim_from: Y-axis lower limit.
    :param y_lim_to: Y-axis upper limit.
    :param hidden_variables: List of hidden variables.
    """
    plt.figure(figsize=(8, 5))

    score_functions = {
        "Total": compute_total_coherency_rate,
        "Faithfulness": compute_faithfulness_coherency_rate,
        "Markov": compute_markov_coherency_rate
    }

    results = generate_average_coherency_scores(sample_sizes_list, num_repetitions, model, score_functions,
                                                hidden_variables)

    markers = {"Total": "s", "Markov": "o", "Faithfulness": "^"}

    for name, (y_vals, _) in results.items():
        plt.scatter(sample_sizes_list, y_vals, s=30, marker=markers[name], label=name)

    plt.xlabel("Sample Size")
    plt.ylabel("Average Coherency Score")
    plt.ylim(y_lim_from, y_lim_to)
    plt.title(title)
    plt.legend()
    plt.savefig(f"plots/{title}.png")


def plot_total_coherency_scores_for_different_models(sample_sizes_list, num_repetitions, models, model_names, title,
                                                     y_lim_from=0, y_lim_to=1.5, hidden_variables=None):
    """
    Plots total coherency scores for multiple models.

    :param sample_sizes_list: List of sample sizes.
    :param num_repetitions: Number of repetitions per sample size.
    :param models: List of models to compare.
    :param model_names: Corresponding names for models.
    :param title: Title of the plot.
    :param y_lim_from: Y-axis lower limit.
    :param y_lim_to: Y-axis upper limit.
    :param hidden_variables: List of hidden variables.
    """
    plt.figure(figsize=(8, 5))

    for model, model_name in zip(models, model_names):
        x_vals = sample_sizes_list
        y_vals, standard_deviations = generate_average_coherency_scores(
            sample_sizes_list, num_repetitions, model, {"Total": compute_total_coherency_rate}, hidden_variables
        )["Total"]

        plt.scatter(x_vals, y_vals, s=30, marker='s', label=model_name)

    plt.xlabel("Sample Size")
    plt.ylabel("Average Coherency Score")
    plt.ylim(y_lim_from, y_lim_to)
    plt.title(title)
    plt.legend()
    plt.savefig(f"plots/{title}.png")
