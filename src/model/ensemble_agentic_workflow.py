from model_proxy.openai_proxy import OpenAIProxy
from model_proxy.anthropic_proxy import AnthropicProxy
from .model import Model
from utils.utils import translate_to_english
from utils.utils import clean_tweet
import pandas as pd
import numpy as np
from scipy.special import softmax
import os

class AWEnsemble(Model):
    def __init__(self, model_config):
        self.task_type = model_config['task_type']
        self.prompts = model_config['prompts']
        self.openai = OpenAIProxy()
        self.anthropic = AnthropicProxy()
        self.model_name = model_config['model_name']
        # self.num_itr = model_config['num_iteration']
        self.translate = model_config['translate']
        self.clean = model_config['clean']
        self.base_models_outputs = []
        
    def train(self, training_config):
        raise RuntimeError('train is not defined for AWClassification')
    
    def save(self, parameters_config):
        raise RuntimeError('save is not defined for AWClassification')
    
    def load(self, parameters_config):
        directory = parameters_config['base_models_directory']
        best_model_outputs = None
        for filename in os.listdir(directory):            
            file = os.path.join(directory, filename)      
            with open(file, 'r') as f:
                labels = []
                for line in f.readlines():
                    labels.append(line.strip().split("\t")[1])
                # remove "Labels" from the first line
                labels = labels[1:]
                if filename.startswith("ExplainZeroShotGPT"): # best model
                    best_model_outputs = labels                                        
                else:
                    self.base_models_outputs.append(labels)
        if best_model_outputs is not None:
            self.base_models_outputs.append(best_model_outputs) # best_model_outputs is self.base_models_outputs[-1]

    def single_instance_inference(self, instance):
        tweet = instance['Texts']
        if self.translate:
            tweet = translate_to_english(self.openai, tweet)
        if self.clean:
            tweet = clean_tweet(tweet)
        
        if self.model_name.startswith("gpt"):
            # use GPT4
            role_content = []
            role_content.append(('system', self.prompts['system_prompt'].format(emotion1=self.base_models_outputs[0][instance.name],\
                emotion2=self.base_models_outputs[1][instance.name], emotion3=self.base_models_outputs[2][instance.name],\
                    emotion4=self.base_models_outputs[3][instance.name], emotion5=self.base_models_outputs[4][instance.name],
                    emotion6=self.base_models_outputs[5][instance.name])))
            role_content.append(('user', self.prompts['user_prompt_template'].format(tweet=tweet)))
            # get predicted_label
            response = self.openai.call_chat_completion_api(role_content, model_name=self.model_name)
        else:
            # use anthropic
            role_content = []
            self.anthropic.set_system_promt(self.prompts['system_prompt'].format(emotion1=self.base_models_outputs[0][instance.name],\
                emotion2=self.base_models_outputs[1][instance.name], emotion3=self.base_models_outputs[2][instance.name],\
                    emotion4=self.base_models_outputs[3][instance.name], emotion5=self.base_models_outputs[4][instance.name],
                    emotion6=self.base_models_outputs[5][instance.name]))
            role_content.append(('user', self.prompts['user_prompt_template'].format(tweet=tweet)))
            # get predicted_label
            response = self.anthropic.call_message_api(role_content, model_name=self.model_name)
        
        response_split = response.split('||')
        instance['Labels'] = response_split[-1].strip().strip('.')
        instance['Explanation'] = response_split[0].strip()
        print('predicted label: ', instance['Labels'])
        print('explanation: ', instance['Explanation'])
        return instance[["Labels", "Explanation"]]

    def inference(self, inference_config):
        test_data_filename = inference_config['test_data_filename']
        output_filename = inference_config['output_filename']
        raw_filename = inference_config['raw_filename']
        test_data = pd.read_csv(test_data_filename, sep='\t')
        if 'Labels' in test_data:
            test_data['gold'] = test_data['Labels']
        test_data[["Labels", "Explanation"]] = test_data.apply(lambda x: self.single_instance_inference(x), axis=1)
        # Iterate over each label in the test_data["Labels"]
        for idx, label in test_data["Labels"].items():
            # Check if the label is legal
            if label not in ["Love", "Joy", "Anger", "Fear", "Sadness", "Neutral"]:
                # Replace illegal labels with the label from the best model at the same index
                test_data.at[idx, "Labels"] = self.base_models_outputs[-1][idx] 
        if 'gold' in test_data:
            test_data[['ID', 'gold', 'Labels']].to_csv(output_filename, sep='\t', index=False)
        else:
            test_data[['ID', 'Labels']].to_csv(output_filename, sep='\t', index=False)
            test_data[['ID', 'Labels', 'Explanation']].to_csv(raw_filename, sep='\t', index=False)
        
        
    ### some failed trials    
    # def load(self, parameters_config):
    #     raise RuntimeError('load is not defined for AWClassification')
    
    # def single_instance_inference(self, instance):
    #     tweet = instance['Texts']
    #     curr_distribution = {"Love":0, "Joy":0, "Anger":0, "Fear":0, "Sadness":0, 'Neutral':0}
    #     for _ in range(self.num_itr):
    #         # input to the LLM
    #         role_content = []
    #         role_content.append(('system', self.prompts['iteration_system_prompt']))
    #         role_content.append(('user', self.prompts['user_prompt_template'].format(tweet=tweet)))
    #         # get predicted_label
    #         predicted_label = self.openai.call_chat_completion_api(role_content, model_name=self.openai_model_name, max_tokens=2)
    #         if predicted_label in curr_distribution:
    #             curr_distribution[predicted_label] += 1
    #     # turn counts into probabilities (for task 2-3)
    #     # Extract values from the dictionary
    #     values = np.array(list(curr_distribution.values()))
    #     # Apply softmax to the values
    #     probabilities = softmax(values)
    #     # Create a new dictionary with the probabilities
    #     prob_distribution = {key: prob for key, prob in zip(curr_distribution.keys(), probabilities)}
    #     # Find the maximum count
    #     max_count = max(curr_distribution.values())
    #     # Find all labels associated with the maximum count
    #     max_labels = [label for label, count in curr_distribution.items() if count == max_count]
        
    #     if len(max_labels) > 1: # there's a tie
    #         # input to the LLM
    #         role_content = []
    #         role_content.append(('system', self.prompts['choose_system_prompt'].format(emotions=max_labels)))
    #         role_content.append(('user', self.prompts['user_prompt_template'].format(tweet=tweet)))
    #         # get predicted_label
    #         predicted_label = self.openai.call_chat_completion_api(role_content, model_name=self.openai_model_name, max_tokens=2) # best label
    #         return predicted_label, prob_distribution
    #     else:    
    #         predicted_label = max_labels[0] # best label
    #         return predicted_label, prob_distribution
        
    # def inference(self, inference_config):
    #     test_data_filename = inference_config['test_data_filename']
    #     output_filename = inference_config['output_filename']
    #     test_data = pd.read_csv(test_data_filename, sep='\t')
    #     if 'Labels' in test_data:
    #         test_data['gold'] = test_data['Labels']
    #     test_data["Labels"], test_data["Distributions"] = zip(*test_data.apply(lambda x: self.single_instance_inference(x), axis=1))
    #     if 'gold' in test_data:
    #         test_data[['ID', 'gold', 'Labels']].to_csv(output_filename, sep='\t', index=False)
    #     else:
    #         test_data[['ID', 'Labels']].to_csv(output_filename + ".tsv", sep='\t', index=False)    
    #         test_data[['ID', 'Distributions']].to_csv(output_filename + "_distributions_only.tsv", sep='\t', index=False)
    #         test_data[['ID', 'Labels', 'Distributions']].to_csv(output_filename + "_raw.tsv", sep='\t', index=False) 