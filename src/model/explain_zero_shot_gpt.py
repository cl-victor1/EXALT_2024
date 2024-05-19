from model_proxy.openai_proxy import OpenAIProxy
from model_proxy.anthropic_proxy import AnthropicProxy
import pandas as pd
from .model import Model

class ExplainZeroShotGPT(Model):
    def __init__(self, model_config):
        self.task_type = model_config['task_type']
        self.prompts = model_config['prompts']
        self.openai = OpenAIProxy()
        self.anthropic = AnthropicProxy()
        #self.openai.set_system_promt(self.prompts['system_prompt'])
        self.openai_model_name = model_config['openai_model_name']
        self.anthropic_model_name = model_config['anthropic_model_name']
        
    def train(self, training_config):
        raise RuntimeError('train is not defined for ExplainZeroShotGPT')
    
    def load(self, parameters_config):
        raise RuntimeError('load is not defined for ExplainZeroShotGPT')

    def save(self, parameters_config):
        raise RuntimeError('save is not defined for FewShotGPT')
    
    def single_instance_inference(self, instance):
        role_content = []
        role_content.append(('system', self.prompts['init_system_prompt']))
        role_content.append(('user', self.prompts['user_prompt_template'].format(tweet=instance['Texts'])))
        prediction = self.openai.call_chat_completion_api(role_content, model_name=self.openai_model_name)
        prediction_split = prediction.split('||')
        instance['Labels'] = prediction_split[-1].strip().strip('.')
        instance['Explanation'] = '||'.join(prediction_split[:-1])
        print(instance.name, 'tweet: ', instance['Texts'])
        print('predicted label: ', instance['Labels'])
        print('explanation: ', instance['Explanation'])
        instance['InitialLabels'] = 'ONLY_CHECKED_ONCE_NO_INITIAL_LABEL'
        instance['InitialExplanation'] = 'ONLY_CHECKED_ONCE_NO_INITIAL_EXPLANATION'
        if 'Neutral' in instance['Labels']:
            print('predicted label was Neutral, double checking')
            instance['InitialLabels'] = 'Neutral'
            instance['InitialExplanation'] = instance['Explanation']
            role_content = []
            #role_content.append(('system', self.prompts['dc_system_prompt']))
            self.anthropic.set_system_promt(self.prompts['dc_system_prompt'])
            role_content.append(('user', self.prompts['user_prompt_template'].format(tweet=instance['Texts'])))
            dc_prediction = self.anthropic.call_message_api(role_content, model_name=self.anthropic_model_name)
            dc_prediction_split = dc_prediction.split('||')
            instance['Labels'] = dc_prediction_split[-1].strip().strip('.')
            instance['Explanation'] ='||'.join(dc_prediction_split[:-1])
            print('double check predicted label: ', instance['Labels'])
            print('double check explanation: ', instance['Explanation'])
        return instance[["Labels", "InitialLabels", "Explanation", "InitialExplanation"]]
    
    def single_instance_inference_2(self, instance):
        role_content = []
        role_content.append(('system', self.prompts['init_system_prompt']))
        role_content.append(('user', self.prompts['user_prompt_template'].format(tweet=instance['Texts'])))
        prediction = self.openai.call_chat_completion_api(role_content, model_name=self.openai_model_name)
        prediction_split = prediction.split('||')
        instance['InitialLabels'] = prediction_split[-1].strip().strip('.')
        instance['InitialExplanation'] = '||'.join(prediction_split[:-1])
        print(instance.name, 'tweet: ', instance['Texts'])
        print('predicted label: ', instance['InitialLabels'])
        print('explanation: ', instance['InitialExplanation'])
        print('Double checking')
        role_content = []
        self.anthropic.set_system_promt(self.prompts['dc_all_system_prompt'].format(emotion=instance['InitialLabels']))
        role_content.append(('user', self.prompts['user_prompt_template'].format(tweet=instance['Texts'])))
        dc_prediction = self.anthropic.call_message_api(role_content, model_name=self.anthropic_model_name)
        dc_prediction_split = dc_prediction.split('||')
        instance['Labels'] = dc_prediction_split[-1].strip().strip('.')
        instance['Explanation'] ='||'.join(dc_prediction_split[:-1])
        print('double check predicted label: ', instance['Labels'])
        print('double check explanation: ', instance['Explanation'])
        return instance[["Labels", "InitialLabels", "Explanation", "InitialExplanation"]]
    
    def single_instance_inference_trigger_word(self, instance):
        role_content = []
        role_content.append(('system', self.prompts['init_system_prompt']))
        role_content.append(('user', self.prompts['user_prompt_template'].format(tweet=instance['Texts'])))
        prediction = self.openai.call_chat_completion_api(role_content, model_name=self.openai_model_name)
        prediction_split = prediction.split('||')
        instance['Labels'] = prediction_split[-1].strip().strip('.')
        instance['Explanation'] = '||'.join(prediction_split[:-1])
        print(instance.name, 'tweet: ', instance['Texts'])
        print('predicted label: ', instance['Labels'])
        print('explanation: ', instance['Explanation'])
        return instance[["Labels", "Explanation"]]

    def single_instance_inference_3(self, instance):
        role_content = []
        self.anthropic.set_system_promt(self.prompts['init_system_prompt'])
        role_content.append(('user', self.prompts['user_prompt_template'].format(tweet=instance['Texts'])))
        prediction = self.anthropic.call_message_api(role_content, model_name=self.anthropic_model_name)
        prediction_split = prediction.split('||')
        instance['InitialLabels'] = prediction_split[-1].strip().strip('.')
        instance['InitialExplanation'] = '||'.join(prediction_split[:-1])
        print(instance.name, 'tweet: ', instance['Texts'])
        print('predicted label: ', instance['InitialLabels'])
        print('explanation: ', instance['InitialExplanation'])
        print('Double checking')
        role_content = []
        role_content.append(('system', self.prompts['dc_all_system_prompt'].format(emotion=instance['InitialLabels'])))
        role_content.append(('user', self.prompts['user_prompt_template'].format(tweet=instance['Texts'])))
        dc_prediction = self.openai.call_chat_completion_api(role_content, model_name=self.openai_model_name)
        dc_prediction_split = dc_prediction.split('||')
        instance['Labels'] = dc_prediction_split[-1].strip().strip('.')
        instance['Explanation'] ='||'.join(dc_prediction_split[:-1])
        print('double check predicted label: ', instance['Labels'])
        print('double check explanation: ', instance['Explanation'])
        return instance[["Labels", "InitialLabels", "Explanation", "InitialExplanation"]]

    def inference(self, inference_config):
        test_data_filename = inference_config['test_data_filename']
        output_filename = inference_config['output_filename']
        test_data = pd.read_csv(test_data_filename, sep='\t')
        #test_data = test_data[:3]
        #test_data = test_data[-6:-5]
        if 'Labels' in test_data:
            test_data['gold'] = test_data['Labels']
        if self.task_type == 'classification':
            # using single_instance_inference as we only need to double check on neutral
            test_data[["Labels", "InitialLabels", "Explanation", "InitialExplanation"]] = test_data.apply(lambda x: self.single_instance_inference(x), axis=1)
        elif self.task_type == 'trigger_detection':
             test_data[["Labels", "Explanation"]] = test_data.apply(lambda x: self.single_instance_inference_trigger_word(x), axis=1)
        if 'gold' in test_data:
            test_data[['ID', 'gold', 'Labels']].to_csv(output_filename, sep='\t', index=False)
        else:
            if self.task_type == 'classification':
                test_data[['ID', 'Texts', 'Labels', 'InitialLabels', 'Explanation', 'InitialExplanation']].to_csv(output_filename, sep='\t', index=False)
            elif self.task_type == 'trigger_detection':
                test_data[['ID', 'Texts', 'Labels', 'Explanation']].to_csv(output_filename, sep='\t', index=False)
