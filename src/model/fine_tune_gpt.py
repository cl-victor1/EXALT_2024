from model_proxy.openai_proxy import OpenAIProxy
from .model import Model
from utils.utils import translate_to_english
from utils.utils import clean_tweet
import pandas as pd

class FineTuneGPT(Model):
    def __init__(self, model_config):
        self.task_type = model_config['task_type']
        self.prompts = model_config['prompts']
        self.openai = OpenAIProxy()
        self.openai.set_system_promt(self.prompts['system_prompt'])
        self.openai_model_name = model_config['openai_model_name']
        self.translate = model_config['translate']
        self.clean = model_config['clean']
    
    def train(self, training_config):
        raise RuntimeError('train has finished')
    
    def load(self, parameters_config):
        raise RuntimeError('nothing to load')
    
    def save(self, parameters_config):
        raise RuntimeError('save is not defined for FineTuneGPT')
    
    def single_instance_inference(self, instance):
        tweet = instance['Texts']
        if self.translate:
            tweet = translate_to_english(self.openai, tweet)
        if self.clean:
            tweet = clean_tweet(tweet)
        role_content = []
        role_content.append(('user', self.prompts['user_prompt_template'].format(tweet=tweet)))
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
                test_data.at[idx, "Labels"] = "Neutral"
                print(idx, label) # logging
        if 'gold' in test_data:
            test_data[['ID', 'gold', 'Labels']].to_csv(output_filename, sep='\t', index=False)
        else:
            test_data[['ID', 'Labels']].to_csv(output_filename, sep='\t', index=False)