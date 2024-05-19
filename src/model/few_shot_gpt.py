from model_proxy.openai_proxy import OpenAIProxy
from model_proxy.anthropic_proxy import AnthropicProxy
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import ast
from .model import Model

'''
prompts = {
    "classification": {
        "system_prompt": "You are a helpful assistant designed to output classification results.",
        "user_prompt_template": "Suppose there are six emotions: Love, Joy, Anger, Fear, Sadness, Neutral. Use your instinct, \
    what is the emotion of the following tweet: '{tweet_text}'. Your response must be just one label from the six labels. \
        Please do not output anything else.",
        "assistant_prompt_template": "{label}"
    },
    "trigger_detection": {
        "system_prompt": "You are a helpful assistant designed to output token classification results.",
        "user_prompt_template": "Treat each space-separated text entity as a token. Punctuations and emojis could also be tokens.\
              For exmpale, thre are 4 tokens in the following sentence 'I'm also happy .' A token is an emotion trigger token if \
                it is used to express the emotion. For example, the token 'happy' is the emotion trigger token in the following sentence \
                    'I'm also happy .' Use your instinct, what are the emotion trigger tokens of the following tweet: '{tweet_text}'. \
                        Please output the value in vector format, where 0 represnets not an emotion trigger token, 1 \represents an emotion \
                            trigger token. The length of the vector should be exactly the same as that of the number of tokens in the tweet. \
                                For example, you should output [0, 0, 1, 0] for the following sentense 'I'm also happy .' \
                                    Please do not output anything else other than the vector itself!",
        "assistant_prompt_template": "{label}"
    }
}
'''

class FewShotGPT(Model):
    def __init__(self, model_config):
        self.task_type = model_config['task_type']
        self.prompts = model_config['prompts']
        self.openai = OpenAIProxy()
        self.anthropic = AnthropicProxy()
        self.openai.set_system_promt(self.prompts['system_prompt'])
        self.anthropic.set_system_promt(self.prompts['system_prompt'])
        if 'openai_model_name' in model_config:
            self.openai_model_name = model_config['openai_model_name']
        if 'anthropic_model_name' in model_config:
            self.anthropic_model_name = model_config['anthropic_model_name']
        self.training_data = None
        self.num_shot = model_config['num_shot']
        if self.num_shot > 0:
            self.knn = KNeighborsClassifier(n_neighbors=self.num_shot)
        else:
            self.knn = None
        self.random_sample = False
        if 'random_sample' in model_config:
            self.random_sample = model_config['random_sample']
        self.instances = None
        self.labels = None
        self.int_labels = None
        
    def train(self, training_config):
        raise RuntimeError('train is not defined for FewShotGPT')
    
    def load(self, parameters_config):
        training_data_filename = parameters_config['training_data_filename']
        if self.num_shot > 0:
            self.training_data = pd.read_csv(training_data_filename, sep='\t')
            self.training_data["embedding"] = self.training_data.embedding.apply(lambda x: ast.literal_eval(x))
            self.instances = np.array(self.training_data["embedding"].tolist())
            self.labels = np.array(self.training_data["Labels"].tolist())
            _, self.int_labels = np.unique(self.labels, return_inverse=True)
            self.knn.fit(self.instances, self.labels)
        else:
            # nothing to load for zero shot
            pass

    def save(self, parameters_config):
        raise RuntimeError('save is not defined for FewShotGPT')
    
    def find_knn(self, instances):
        return self.knn.kneighbors(instances, n_neighbors=self.num_shot, return_distance=False)
    
    def single_instance_inference(self, instance, knn_indices=None):
        role_content = []
        if knn_indices is not None:
            for _, row in self.training_data[['Texts', 'Labels']].loc[knn_indices[instance.name]].iterrows():
                role_content.append(('user', self.prompts['user_prompt_template'].format(tweet_text=row['Texts'])))
                role_content.append(('assistant', self.prompts['assistant_prompt_template'].format(label=row['Labels'])))
        role_content.append(('user', self.prompts['user_prompt_template'].format(tweet_text=instance['Texts'])))
        predicted_label = self.openai.call_chat_completion_api(role_content, model_name=self.openai_model_name)
        #predicted_label = self.anthropic.call_message_api(role_content, model_name=self.anthropic_model_name)
        print(instance.name, 'tweet: ', instance['Texts'])
        print('predicted label: ', predicted_label)
        return predicted_label

    def inference(self, inference_config):
        test_data_filename = inference_config['test_data_filename']
        output_filename = inference_config['output_filename']
        test_data = pd.read_csv(test_data_filename, sep='\t')
        if 'Labels' in test_data:
            test_data['gold'] = test_data['Labels']
        if self.num_shot > 0:
            test_data["embedding"] = test_data.embedding.apply(lambda x: np.array(ast.literal_eval(x)))
            test_instances = np.array(test_data["embedding"].tolist())
            if self.random_sample:
                knn_indices = np.random.randint(0, high=len(self.training_data.Texts), size=(len(test_data.Texts), self.num_shot))
            else:
                knn_indices = self.find_knn(test_instances)
            assert len(test_data.Texts) == len(knn_indices)
            test_data["Labels"] = test_data.apply(lambda x: self.single_instance_inference(x, knn_indices=knn_indices), axis=1)
        else:
            test_data["Labels"] = test_data.apply(lambda x: self.single_instance_inference(x), axis=1)
        if 'gold' in test_data:
            test_data[['ID', 'gold', 'Labels']].to_csv(output_filename, sep='\t', index=False)
        else:
            test_data[['ID', 'Labels']].to_csv(output_filename, sep='\t', index=False)