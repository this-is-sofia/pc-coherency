import json
import os


def load_result_dataset(results_directory) -> dict:
    """ Load the result datasets from the given directory and return them as a dictionary.
    """
    datasets = {}

    # iterate through the result folder
    for folder in os.listdir(results_directory):
        if os.path.isdir(f"{results_directory}/{folder}"):
            datasets[folder] = {}
            for file in os.listdir(f"{results_directory}/{folder}"):
                if file.endswith(".json"):
                    with open(f"{results_directory}/{folder}/{file}", "r") as f:
                        datasets[folder][file.replace(".json", "")] = json.load(f)

    return datasets


def generate_result_table(result, name="") -> str:
    """generate latex table from the given result dictionary.
    """
    dimensions = result["sample_sizes"]
    rows = ["Total", "Faithfulness", "Markov", "Orientation Conflicts"]
    result = result["scores"]

    table = """
\\begin{table}
\centering
\caption{"""+name+"""}\label{tab:data}"""
    accuracy = 3
    table += "\\begin{tabular}{"+"c"*(len(dimensions)+1)+"}\n"
    table += "\\toprule % from booktabs package\n"
    table += "\\bfseries & " +" & ".join(["\\bfseries " + str(d) for d in dimensions]) + " \\\\ \n"
    table += "\midrule % from booktabs package\n"
    for row in rows:
        if len(result[row]) == 1:
            table += row + " & " + " & ".join([str(round(result[row][0][i], accuracy)) for i in range(len(dimensions))]) + " \\\ \n"
        else:
            table += row + " & " + " & ".join([str(round(result[row][0][i], accuracy)) + " (" + str(round(result[row][1][i], accuracy))+") " for i in range(len(dimensions))]) + " \\\ \n"
    table += "\\bottomrule % from booktabs package\n"
    table += "\\end{tabular}\n"
    table += "\\end{table}\n"

    return table




if __name__ == "__main__":
    results = load_result_dataset("results")
    for dataset in results:
        for result in results[dataset]:
            table = generate_result_table(results[dataset][result], f"{dataset} {result}".replace("_", " "))
            print(table)
            print("\n\n")
