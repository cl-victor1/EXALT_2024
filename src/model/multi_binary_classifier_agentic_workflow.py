from model_proxy.openai_proxy import OpenAIProxy
from .model import Model
from utils.utils import translate_to_english
from utils.utils import clean_tweet
import pandas as pd

class MBCAgenticWorkflow(Model):
    def __init__(self, model_config):
        self.task_type = model_config['task_type']
        self.prompts = model_config['prompts']
        self.emotions = model_config['emotions']
        self.openai = OpenAIProxy()
        self.openai_model_name = model_config['openai_model_name']
        self.stats = {'all_negative': 0, 'one_positive': 0, 'agentic_workflow': 0}
        
    def train(self, training_config):
        raise RuntimeError('train is not defined for MBCAgenticWorkflow')
    
    def load(self, parameters_config):
        raise RuntimeError('load is not defined for MBCAgenticWorkflow')

    def save(self, parameters_config):
        raise RuntimeError('save is not defined for MBCAgenticWorkflow')

    def single_instance_inference(self, instance):
        tweet = instance['Texts']
        emotions = set()
        for emotion in self.emotions:
            role_content = []
            role_content.append(('system', self.prompts['bc_system_prompt'].format(emotion=emotion)))
            role_content.append(('user', self.prompts['user_prompt_template'].format(tweet=tweet)))
            yes_or_no = self.openai.call_chat_completion_api(role_content, model_name=self.openai_model_name)
            if yes_or_no.lower() == 'yes':
                emotions.add(emotion)
        if len(emotions) == 0:
            role_content = []
            role_content.append(('system', self.prompts['dc_system_prompt'].format(emotions=self.emotions)))
            role_content.append(('user', self.prompts['user_prompt_template'].format(tweet=tweet)))
            predicted_emition = self.openai.call_chat_completion_api(role_content, model_name=self.openai_model_name)
            print(instance.name, 'tweet: ', instance['Texts'])
            print('predicted label: ', predicted_emition)
            self.stats['all_negative'] += 1
            return predicted_emition
        elif len(emotions) == 1:
            print(instance.name, 'tweet: ', instance['Texts'])
            print('predicted label: ', list(emotions)[0])
            self.stats['one_positive'] += 1
            return list(emotions)[0]
        else:
            role_content = []
            role_content.append(('system', self.prompts['af_system_prompt'].format(emotions=list(emotions))))
            role_content.append(('user', self.prompts['user_prompt_template'].format(tweet=tweet)))
            predicted_emition = self.openai.call_chat_completion_api(role_content, model_name=self.openai_model_name)
            print(instance.name, 'tweet: ', instance['Texts'])
            print('predicted label: ', emotion)
            self.stats['agentic_workflow'] += 1
            return predicted_emition

    def inference(self, inference_config):
        test_data_filename = inference_config['test_data_filename']
        output_filename = inference_config['output_filename']
        test_data = pd.read_csv(test_data_filename, sep='\t')
        if 'Labels' in test_data:
            test_data['gold'] = test_data['Labels']   
        test_data["Labels"] = test_data.apply(lambda x: self.single_instance_inference(x), axis=1)
        if 'gold' in test_data:
            test_data[['ID', 'gold', 'Labels']].to_csv(output_filename, sep='\t', index=False)
        else:
            test_data[['ID', 'Labels']].to_csv(output_filename, sep='\t', index=False)
        print(self.stats)