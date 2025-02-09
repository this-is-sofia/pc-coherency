import os

from plotting import plot_coherency_scores_test

script_dir = os.path.dirname(os.path.abspath(__file__))
folder_results = os.path.join(script_dir, "results")

plot_coherency_scores_test(results_folder=folder_results, output_folder="plots")