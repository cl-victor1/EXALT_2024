import argparse
import json
from model.few_shot_gpt import FewShotGPT
from model.simple_knn import SimpleKNN
from model.classification_agentic_workflow import AWClassification
from model.ensemble_agentic_workflow import AWEnsemble
from model.trigger_agentic_workflow import AWTrigger
from model.fine_tune_gpt import FineTuneGPT
from model.multi_binary_classifier_agentic_workflow import MBCAgenticWorkflow
from model.bert_knn import BERTKNN
from model.explain_zero_shot_gpt import ExplainZeroShotGPT
#from model.naivebayes_embedding_toy import NaiveBayesEmbeddingToy
def main():
    models = [FewShotGPT, SimpleKNN, AWClassification, FineTuneGPT, MBCAgenticWorkflow, AWTrigger, BERTKNN, ExplainZeroShotGPT, AWEnsemble] #, NaiveBayesEmbeddingToy]
    parser = argparse.ArgumentParser(description='Model Inference')
    parser.add_argument('-mn', '--model_name', choices=[m.__name__ for m in models], required=True)
    parser.add_argument('-mc', '--model_config', required=True)
    parser.add_argument('-pc', '--parameters_config')
    parser.add_argument('-ic', '--inference_config', required=True)
    args = parser.parse_args()
    with open(args.model_config, 'r') as mc:
        model_config = json.load(mc)
    with open(args.inference_config, 'r') as ic:
        inference_config = json.load(ic)
    if args.parameters_config is not None:
        with open(args.parameters_config, 'r') as pc:
            parameters_config = json.load(pc)
    # initialize model
    for m in models:
        if m.__name__ == args.model_name:
            model = m(model_config)
            break
    # if model_parameters is set, load model parameters
    if args.parameters_config is not None:
        model.load(parameters_config)

    #''' added by sheng'''
    #if args.model_name == 'NaiveBayesEmbeddingToy':
    #    model.train(model_config)
    #    model.inference_m1(inference_config, arg_run_using_train = False)
    #else:
    #    pass;
    #''' added by sheng'''
    model.inference(inference_config)

if __name__ == '__main__':
    main()
