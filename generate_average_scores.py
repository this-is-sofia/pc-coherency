import json
import os

import numpy as np
from causy.causal_discovery.constraint.algorithms.pc import PCClassic, PC

from utils import _get_number_of_orientation_conflicts


def generate_test_data_and_run_pc(model, sample_size, hidden_variables=None, file_name="no_name", repitition=1, folder_name="no_folder_name"):
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

    tst = PC()
    tst.create_graph_from_data(test_data)
    tst.create_all_possible_edges()
    tst.execute_pipeline_steps()

    # Dump data to JSON – uncomment if needed

    test_data_for_dumping = {}
    for key in test_data.keys():
        test_data_for_dumping[key] = test_data[key].tolist()

    # !attention!
    # uncomment the following lines to save all data sets for reproducibility,
    # but the number of files may be very large:
    # number of models * number of sample sizes * number of repetitions * number different versions

    # if not os.path.exists("data_sets"):
    #     os.makedirs("data_sets")
    #folder = f"data_sets/{folder_name}_{file_name}_sample_size_{sample_size}"
    #if not os.path.exists(folder):
    #    os.makedirs(folder)
    #with open(f"{folder}/repitition_{repitition}.json", "w") as f:
    #    json.dump(test_data_for_dumping, f, indent=4)

    return tst


def generate_average_coherency_scores(sample_sizes, num_repetitions, model, score_functions, hidden_variables=[],
                                      model_name="no_name", folder_name="no_folder_name"):
    """
    Computes the average and standard deviation of coherency scores using given scoring functions.
    Also records the average number of orientation conflicts.

    :param sample_sizes: List of sample sizes.
    :param num_repetitions: Number of repetitions per sample size.
    :param model: The model to generate data.
    :param score_functions: Dictionary with {score_name: score_function}.
    :param hidden_variables: List of hidden variables to exclude from test data.
    :return: Dictionary {score_name: (averages, standard_deviations)}
    """
    results = {name: ([], []) for name in score_functions.keys()}  # Initialize lists
    orientation_conflicts = []  # Store average orientation conflicts per sample size

    # Construct the file name dynamically based on parameters
    hidden_var_str = "_".join(hidden_variables) if hidden_variables else "no_hidden_vars"
    model_name = model_name.replace(" ", "_")
    file_name = f"{model_name}_{hidden_var_str}"
    folder = f"results/{folder_name}"

    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = f"{folder}/{file_name}.json"

    for sample_size in sample_sizes:
        scores = {name: [] for name in score_functions.keys()}  # Store scores per iteration
        conflicts = []  # Store orientation conflicts per repetition

        print(model_name, hidden_variables, sample_size)

        for repetition in range(num_repetitions):
            tst = generate_test_data_and_run_pc(
                model,
                sample_size,
                hidden_variables,
                file_name,
                repetition,
                folder_name
            )

            for name, func in score_functions.items():
                scores[name].append(func(tst))

            # Compute and store orientation conflicts
            conflicts.append(_get_number_of_orientation_conflicts(tst.graph.action_history))

        for name in score_functions.keys():
            results[name][0].append(np.mean(scores[name]))  # Average
            results[name][1].append(np.std(scores[name]))  # Standard deviation

        # Store average orientation conflicts for this sample size
        orientation_conflicts.append(np.mean(conflicts))

    results["Orientation Conflicts"] = [orientation_conflicts]

    # Save results including orientation conflicts
    results_with_sample_sizes_and_average_number_of_orientation_conflicts = {
        "sample_sizes": sample_sizes,
        "scores": results,
    }

    with open(file_path, "w") as f:
        json.dump(results_with_sample_sizes_and_average_number_of_orientation_conflicts, f, indent=4)

    return results
