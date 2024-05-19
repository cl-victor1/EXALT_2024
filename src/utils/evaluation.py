import numpy as np
import ast
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score

def read_classification_output(classification_output_file, delimiter='\t'):
    """Read from the classification output file.

    Args:
        classification_output_file: The classification output file name.
        delimiter: The delimiter used in the file. Default: \t

    Returns:
        gold: A list of gold labels from the classification output file.
        prediction: A list of prediction labels from the classification output file.
    """
    gold = []
    prediction = []
    with open(classification_output_file, 'r') as f:
        count = 0
        for line in f:
        # skip header line
            if count > 0:
                splitted_line = line.strip().split(delimiter)
                gold.append(splitted_line[1])
                prediction.append(splitted_line[2])
            count += 1
    return gold, prediction

def calculate_accuracy(gold, prediction, to_stdout=True):
    """Calculate the accuracy.

    Args:
        gold: A list of gold labels.
        prediction: A list of prediction labels.
        to_stdout: If True, print the calculated accuracy to standard out.
    
    Returns:
        accuracy: The calculated accuracy.
    """
    assert len(gold) == len(prediction)
    correct_count = 0
    for i in range(len(gold)):
        if gold[i] == prediction[i]:
            correct_count += 1
    accuracy = correct_count / len(gold)
    if to_stdout:
        print(f'The accuracy of the classification is {accuracy}')
    return accuracy

def calculate_per_class_prfs(labels, gold, prediction, to_stdout=True):
    """Calculate the per class precision, recall, f1 score and support.

    Args:
        labels: A list of valid labels.
        gold: A list of gold labels.
        prediction: A list of prediction labels.
        to_stdout: If True, print the calculated precision recall f1 score and support to standard out.
    
    Returns:
        per_class_prfs: The calculated per class precision, recall, f1 score and support.
    """
    # labels = ['Love', 'Joy', 'Anger', 'Fear', 'Sadness', 'Neutral']
    prfs = precision_recall_fscore_support(np.array(gold), np.array(prediction), labels=labels)
    per_class_prfs = dict()
    for i in range(len(labels)):
        per_class_prfs[labels[i]] = {'precision': prfs[0][i], 'recall': prfs[1][i], 'f1': prfs[2][i], 'support': prfs[3][i]}
    if to_stdout:
        for label in labels:
            print(f'class {label}:')
            print(f'precision: {per_class_prfs[label]["precision"]}')
            print(f'recall: {per_class_prfs[label]["recall"]}')
            print(f'f1: {per_class_prfs[label]["f1"]}')
            print(f'support: {per_class_prfs[label]["support"]}')
    return per_class_prfs

def calculate_prf(labels, gold, prediction, average='macro', to_stdout=True):
    """Calculate the averaged precision, recall, f1 score.

    Args:
        labels: A list of valid labels.
        gold: A list of gold labels.
        prediction: A list of prediction labels.
        average: The average method to use. Default: macro
        to_stdout: If True, print the calculated precision recall f1 score to standard out.
    
    Returns:
        averaged_prf: The calculated averaged precision, recall, f1 score.
    """
    if average == None:
        raise ValueError("The average arg is required. Use calculate_per_class_prfs\
         to calcualte per class precision, recall, f1 score and support")
    averaged_prf = precision_recall_fscore_support(np.array(gold), np.array(prediction), labels=labels, average=average)
    if to_stdout:
        print(f'{average}-averaged precision: {averaged_prf[0]}')
        print(f'{average}-averaged recall: {averaged_prf[1]}')
        print(f'{average}-averaged f1: {averaged_prf[2]}')
    return averaged_prf

def read_trigger_detection_output(trigger_detection_output_file, delimiter='\t'):
    """Read from the trigger detection output file.

    Args:
        trigger_detection_output_file: The trigger detection output file name.
        delimiter: The delimiter used in the file. Default: \t

    Returns:
        gold: A list of gold token labels from the classification output file.
        prediction: A list of prediction token labels from the classification output file.
    """
    gold = []
    prediction = []
    with open(trigger_detection_output_file, 'r') as f:
        count = 0
        for line in f:
            if count > 0:
                splitted_line = line.strip().split(delimiter)
                gold.append(ast.literal_eval(splitted_line[1]))
                prediction.append(ast.literal_eval(splitted_line[2]))
            count += 1
    return gold, prediction

def token_precision(gold, prediction):
    """Compute the token precision on one instance.

    Args:
        gold: A list of gold labels.
        prediction: A list of prediction labels.
    
    Returns:
        Calcualted precision.
    """
    return precision_recall_fscore_support(np.array(gold), np.array(prediction), average='binary')[0]

def token_recall(gold, prediction):
    """Compute the token recall on one instance.

    Args:
        gold: A list of gold labels.
        prediction: A list of prediction labels.
    
    Returns:
        Calcualted recall.
    """
    return precision_recall_fscore_support(np.array(gold), np.array(prediction), average='binary')[1]

def token_f1(gold, prediction):
    """Compute the token f1 score on one instance.

    Args:
        gold: A list of gold labels.
        prediction: A list of prediction labels.
    
    Returns:
        Calcualted f1 score.
    """
    return precision_recall_fscore_support(np.array(gold), np.array(prediction), average='binary')[2]

def token_mean_average_precision(gold, prediction):
    """Compute the mean average precision on one instance.

    Args:
        gold: A list of gold labels.
        prediction: A list of prediction labels.
    
    Returns:
        Calcualted mean average precision.
    """
    ap_pos = average_precision_score(np.array(gold), np.array(prediction), pos_label=1)
    ap_neg = average_precision_score(np.array(gold), np.array(prediction), pos_label=0)
    return (ap_pos + ap_neg) / 2

def apply_across_instances(measure, gold, prediction, to_stdout=True):
    """Apply the specified measure across all instances and take the average.

    Args:
        measure: The measure function to apply
        gold: A list of gold label lists.
        prediction: A list of prediction label lists.
    
    Returns:
        average_score: The average of computed measure across all instances.
    """
    assert len(gold) == len(prediction)
    average_score = np.mean([measure(gold[i], prediction[i]) for i in range(len(gold))])
    if to_stdout:
        print(f'{measure.__name__} across all instances: {average_score}')
    return average_score