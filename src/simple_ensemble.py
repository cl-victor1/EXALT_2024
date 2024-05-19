import pandas as pd
import numpy as np
from collections import Counter
import os
import sys
from scipy.special import softmax

def get_all_files_in_directory(directory, if_weight=False):
    files = []
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        if if_weight:
            weight = 0 # set weight based on F1 of each model on the dev set 
            if filename.startswith("ExplainZeroShotGPT"):
                weight = 0.6
            elif filename.startswith("multi_binary") or filename.startswith("once") or filename.startswith("twice"):
                weight = 0.57
            elif filename.startswith("ZeroShotGPT"):
                weight = 0.54
            elif filename.startswith("FewShotGPT"):
                weight = 0.55
            elif filename.startswith("BERT_dev_results_knn") or filename.startswith("fine_tune"):
                weight = 0.48
            elif filename.startswith("OpenAI_knn_6"):
                weight = 0.42
            else:
                raise RuntimeError("illegal file")
            file = (file, weight)        
        files.append(file)
    return files

def normalize_weights(weights):
    # total_weight = sum(weights.values())
    # return {file: weight / total_weight for file, weight in weights.items()}
    # using softmax, which has the same result as above normalization
    values = np.array(list(weights.values()))
    normalized_values = softmax(values)
    return {file: normalized_values[i] for i, file in enumerate(weights.keys())}

def main(): 
    # List of input files
    input_directory = sys.argv[1]
    output_file = sys.argv[2]
    if_weight = int(sys.argv[3]) # 0 without weights / 1 with weights
    input_files = get_all_files_in_directory(input_directory, if_weight)

    if isinstance(input_files[0], str): # no weights
        # Read each input file and store labels in a dictionary
        label_counts = {}
        for file in input_files:
            df = pd.read_csv(file, sep='\t')
            for index, row in df.iterrows():
                label_counts.setdefault(row['ID'], []).append(row['Labels'])

        # Find the most frequent label for each ID
        most_common_labels = {id: Counter(labels).most_common(1)[0][0] for id, labels in label_counts.items()}

    else: # with weights
        # Assign weights to each input file based on their names
        file_weights = {file[0]: file[1] for file in input_files}  # Initialize with equal weights
        normalized_weights = normalize_weights(file_weights)

        # Read each input file and store labels in a dictionary with weighted counts
        weighted_label_counts = {}
        for file, weight in normalized_weights.items():
            df = pd.read_csv(file, sep='\t')
            for index, row in df.iterrows():
                weighted_label_counts.setdefault(row['ID'], Counter())[row['Labels']] += weight

        # Find the most frequent label for each ID based on weighted counts
        most_common_labels = {id: labels.most_common(1)[0][0] for id, labels in weighted_label_counts.items()}
    
    # Write the output to a new CSV file
    with open(output_file, 'w') as f:
        f.write("ID\tLabels\n")
        for id, label in most_common_labels.items():
            f.write(f"{id}\t{label}\n")

if __name__ == "__main__":
    main()
