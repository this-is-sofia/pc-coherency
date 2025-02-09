import json
import os
import matplotlib.pyplot as plt

# Set default plot style
plt.style.use('seaborn-v0_8-colorblind')

def plot_coherency_scores(results_folder="results", output_folder="plots"):
    """
    Iterates through the results folder, reads each file, and plots the coherency scores.

    :param results_folder: Path to the results directory containing subfolders with result files.
    :param output_folder: Path to save the generated plots.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for folder in os.listdir(results_folder):
        folder_path = os.path.join(results_folder, folder)
        if os.path.isdir(folder_path):

            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.endswith(".json"):  # Assuming results are stored in CSV files

                    # Load the json file
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    sample_sizes = data["sample_sizes"]

                    plt.figure(figsize=(6, 4.5))

                    for key in data["scores"]:
                        if key == "Orientation Conflicts":
                            continue
                        # plot data but with logarithmic scale on x-axis

                        plt.scatter(sample_sizes, data["scores"][key][0], s=10, marker="s", label=key)

                    plt.xscale("log")
                    plt.xlabel("Sample Size")
                    plt.ylabel("Average Coherency Rate")
                    plt.ylim(0.4, 1.05)
                    plt.title(f"{folder}/{file}")
                    plt.legend()

                    # Save the plot
                    plot_filename = f"{folder}_{file.replace('.json', '')}.png"
                    plt.savefig(os.path.join(output_folder, plot_filename))
                    plt.close()