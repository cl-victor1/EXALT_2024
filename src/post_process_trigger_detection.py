import pandas as pd
import ast
import re

def find_sublist_indices(main_list, sublist):
    return [i for i in range(len(main_list)) if main_list[i:i+len(sublist)] == sublist]

def split_on_multiple_chars_and_keep_delimiter(string, delimiters):
    pattern = '|'.join(map(re.escape, delimiters))
    return [substr for substr in re.split(f'({pattern})', string) if substr]

def process(instance):
    text = instance['Texts'].split()
    binary_labels = [0 for _ in range(len(text))]
    #print(instance['Labels'])
    raw_labels = ast.literal_eval(instance['Labels'])
    for raw_label in raw_labels:
        delimiters = [' ', "'", '’', '#', '-']
        raw_label_split = split_on_multiple_chars_and_keep_delimiter(raw_label, delimiters)
        if " " in raw_label_split:
            raw_label_split = [x for x in raw_label_split if x != " "]
        if len(raw_label_split) > 1:
            # hack: assume only one match
            try:
                index = find_sublist_indices(text, raw_label_split)[0]
                for i in range(len(raw_label_split)):
                    binary_labels[index + i] = 1
            except:
                print(instance)
        else:
            try:
                index = text.index(raw_label_split[0])
                binary_labels[index] = 1
            except:
                print(instance)
    #instance['Labels'] = instance['Labels'].strip()
    instance["Labels"] = binary_labels
    return instance["Labels"]

test_data = pd.read_csv('../outputs/evaluation/BinaryTriggers/ExplainZeroShotGPT_gpt4o_processed.tsv', sep='\t')
test_data["Labels"] = test_data.apply(lambda x: process(x), axis=1)
test_data[['ID', 'Labels']].to_csv("../outputs/evaluation/BinaryTriggers/ExplainZeroShotGPT_gpt4o_postprocessed.tsv", sep='\t', index=False)