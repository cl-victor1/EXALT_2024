'''
For Trigger Word Detection task, tweets cannot be preprocessed.
For each tweet:
1. Create a list of 0s. (the length of the list == the number of tokens)
2. The {emotion} of this tweet has been determined using the algorithm for Task 1.
3. Get the trigger words of that {emotion} using agent 1.
4. check/modify the trigger words using agent 2.
5. turn slots correpsonding to trigger words to 1s.
'''
from model_proxy.openai_proxy import OpenAIProxy
from .model import Model
import pandas as pd
from collections import defaultdict

class AWTrigger(Model):
    def __init__(self, model_config):
        self.task_type = model_config['task_type']
        self.prompts = model_config['prompts']
        self.openai = OpenAIProxy()
        self.openai_model_name = model_config['openai_model_name']
        # self.emotion_labels = []
        
    def train(self, training_config):
        raise RuntimeError('train is not defined for AWTrigger')
    
    def load(self, parameters_config):
        # emotion_output = parameters_config['emotion_output']
        # with open(emotion_output, 'r') as f:
        #     for line in f.readlines():
        #         if line.strip(): # skip empty lines
        #             self.emotion_labels.append(line.strip().split("\t")[2])
        # # remove "Labels" from the first line
        # self.emotion_labels = self.emotion_labels[1:]
        raise RuntimeError('load is not defined for AWTrigger')

    def save(self, parameters_config):
        raise RuntimeError('save is not defined for AWTrigger')

    def single_instance_inference(self, instance):
        tweet = instance['Texts']
        # emotion=self.emotion_labels[instance.name] 
        tweet_words = tweet.strip().split()   
        list_triggers = [0 for _ in range(len(tweet_words))] # Create a list of 0s   
        # map words to indices
        words_to_indices = defaultdict(list) 
        for index, word in enumerate(tweet_words):
            words_to_indices[word].append(index)     
        # Agent 1: get trigger words  
        role_content = []
        role_content.append(('system', self.prompts['a1_system_prompt'])) # .format(emotion=emotion)
        role_content.append(('user', self.prompts['user_prompt_template'].format(tweet=tweet)))
        response = self.openai.call_chat_completion_api(role_content, model_name=self.openai_model_name)
        response_split = response.split('||')
        triggers = response_split[-1].strip()
        instance['Explanation'] = response_split[0].strip() # without direct usage
        
        # # Agent 2: revise trigger words 
        # role_content = []
        # role_content.append(('system', self.prompts['a2_system_prompt'].format(triggers=triggers)))
        # role_content.append(('user', self.prompts['user_prompt_template'].format(tweet=tweet)))
        # triggers =  self.openai.call_chat_completion_api(role_content, model_name=self.openai_model_name)        
        
        # update list_triggers
        for trigger in triggers.split():
            try: # avoid unexpected triggers returned by OpenAIProxy
                indices = words_to_indices[trigger]
                for index in indices:
                    list_triggers[index] = 1   
            except:
                continue
        instance["Labels"] = list_triggers
        print(instance.name, 'tweet: ', instance['Texts'])
        print('predicted label: ', list_triggers)
        return instance[["Labels", "Explanation"]]

    def inference(self, inference_config):
        test_data_filename = inference_config['test_data_filename']
        output_filename = inference_config['output_filename']
        raw_filename = inference_config['raw_filename']
        test_data = pd.read_csv(test_data_filename, sep='\t')
        if 'Labels' in test_data: # with gold labels
            test_data['gold'] = test_data['Labels']
        test_data[["Labels", "Explanation"]] = test_data.apply(lambda x: self.single_instance_inference(x), axis=1)
        if 'gold' in test_data: # with gold labels
            test_data[['ID', 'gold', 'Labels']].to_csv(output_filename, sep='\t', index=False)
        else:
            test_data[['ID', 'Labels']].to_csv(output_filename, sep='\t', index=False)
            test_data[['ID', 'Labels', 'Explanation']].to_csv(raw_filename, sep='\t', index=False)
        
