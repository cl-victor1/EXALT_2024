import argparse
from utils.evaluation import read_classification_output
from utils.evaluation import calculate_accuracy
from utils.evaluation import calculate_per_class_prfs
from utils.evaluation import calculate_prf
from utils.evaluation import read_trigger_detection_output
from utils.evaluation import token_precision
from utils.evaluation import token_recall
from utils.evaluation import token_f1
from utils.evaluation import token_mean_average_precision
from utils.evaluation import apply_across_instances

def main():
    parser = argparse.ArgumentParser(description='Evaluate experiement results')
    parser.add_argument('-f', '--filename', required=True)
    parser.add_argument('-t', '--type', choices=['classification', 'trigger_detection'], required=True)
    args = parser.parse_args()
    filename = args.filename
    task_type = args.type
    if task_type == 'classification':
        gold, prediction = read_classification_output(filename)
        labels = ['Love', 'Joy', 'Anger', 'Fear', 'Sadness', 'Neutral']
        calculate_accuracy(gold, prediction)
        calculate_per_class_prfs(labels, gold, prediction)
        calculate_prf(labels, gold, prediction)
    elif task_type == 'trigger_detection':
        gold, prediction = read_trigger_detection_output(filename)
        apply_across_instances(token_precision, gold, prediction)
        apply_across_instances(token_recall, gold, prediction)
        apply_across_instances(token_f1, gold, prediction)
        apply_across_instances(token_mean_average_precision, gold, prediction)

if __name__ == '__main__':
  main()
