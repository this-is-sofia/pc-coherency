import matplotlib.pyplot as plt
import numpy as np
from causy.causal_discovery.constraint.algorithms.pc import PCClassic
from causy.sample_generator import IIDSampleGenerator, SampleEdge, NodeReference
from compute_rates import compute_total_coherency_rate


def generate_coherency_scores(sample_sizes, num_repetitions):
    model_1 = IIDSampleGenerator(
        edges=[
            SampleEdge(NodeReference("X"), NodeReference("Y"), 5),
            SampleEdge(NodeReference("Y"), NodeReference("Z"), 6),
        ],
    )

    average_coherency_rates = []
    standard_deviations = []
    for sample_size in sample_sizes:
        total_coherency_rates = []
        for _ in range(num_repetitions):
            test_data, graph = model_1.generate(sample_size)
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


def plot_coherency_scores(sample_sizes_list, num_repetitions):
    plt.figure(figsize=(8, 5))

    x_vals = sample_sizes_list
    y_vals, standard_deviations = generate_coherency_scores(sample_sizes_list, num_repetitions)

    plt.scatter(x_vals, y_vals, label="Average Coherency Score")
    plt.errorbar(x_vals, y_vals, yerr=standard_deviations, fmt='o', label="Standard Deviation", linestyle='None')

    plt.xlabel("Sample Size")
    plt.ylabel("Average Coherency Score")
    plt.ylim(0, 1)
    plt.title("Coherency Score vs Sample Size")
    plt.legend()
    plt.savefig("coherency_scores.png")


# Example usage
sample_sizes_list = [i for i in range(100, 1000, 100)]
num_repetitions = 50
plot_coherency_scores(sample_sizes_list, num_repetitions)
