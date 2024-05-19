from model_proxy.openai_proxy import OpenAIProxy
from .model import Model
from utils.utils import translate_to_english
from utils.utils import clean_tweet
import pandas as pd

class AWClassification(Model):
    def __init__(self, model_config):
        self.task_type = model_config['task_type']
        self.prompts = model_config['prompts']
        self.openai = OpenAIProxy()
        self.openai_model_name = model_config['openai_model_name']
        self.translate = model_config['translate']
        self.clean = model_config['clean']
        self.model_1_labels = [] 
        self.model_2_labels = [] # model_2 has a better F1-score
        
    def train(self, training_config):
        raise RuntimeError('train is not defined for AWClassification')
    
    def load(self, parameters_config):
        model_1_output = parameters_config['model_1_output']
        model_2_output = parameters_config['model_2_output']
        with open(model_1_output, 'r') as f:
            for line in f.readlines():
                self.model_1_labels.append(line.strip().split("\t")[1])
        with open(model_2_output, 'r') as f:
            for line in f.readlines():
                self.model_2_labels.append(line.strip().split("\t")[1])
        # remove "Labels" from the first line for both
        self.model_1_labels = self.model_1_labels[1:]
        self.model_2_labels = self.model_2_labels[1:]

    def save(self, parameters_config):
        raise RuntimeError('save is not defined for AWClassification')

    def single_instance_inference(self, instance):
        tweet = instance['Texts']
        if self.translate:
            tweet = translate_to_english(self.openai, tweet)
        if self.clean:
            tweet = clean_tweet(tweet)
        # input to the LLM
        role_content = []
        role_content.append(('system', self.prompts['system_prompt'].format(emotion1=self.model_1_labels[instance.name],emotion2=self.model_2_labels[instance.name])))
        role_content.append(('user', self.prompts['user_prompt_template'].format(tweet=tweet)))
        # get predicted_label
        predicted_label = self.openai.call_chat_completion_api(role_content, model_name=self.openai_model_name, max_tokens=2)
        # print(instance.name, 'tweet: ', instance['Texts'])
        # print('predicted label: ', predicted_label)
        return predicted_label

    def inference(self, inference_config):
        test_data_filename = inference_config['test_data_filename']
        output_filename = inference_config['output_filename']
        test_data = pd.read_csv(test_data_filename, sep='\t')
        if 'Labels' in test_data:
            test_data['gold'] = test_data['Labels']
        test_data["Labels"] = test_data.apply(lambda x: self.single_instance_inference(x), axis=1)
        # Iterate over each label in the test_data["Labels"]
        for idx, label in test_data["Labels"].items():
            # Check if the label is legal
            if label not in ["Love", "Joy", "Anger", "Fear", "Sadness", "Neutral"]:
                # Replace illegal labels with the label from model_2_labels at the same index
                test_data.at[idx, "Labels"] = self.model_2_labels[idx]
                print(idx, label) # logging
        if 'gold' in test_data:
            test_data[['ID', 'gold', 'Labels']].to_csv(output_filename, sep='\t', index=False)
        else:
            test_data[['ID', 'Labels']].to_csv(output_filename, sep='\t', index=False)
        
