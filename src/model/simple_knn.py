import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import ast
from .model import Model

'''
Sample model config:
{
    "task_type": "classification",
    "n_neighbors": 5
}

Sample parameters config:
{
    "training_data_filename": "../data/training_data_filename"
}

Sample inference config:
{
    "test_data_filename": "../data/test_data_filename",
    "output_filename": "../experiment/output_filename"
}
'''

class SimpleKNN(Model):
    def __init__(self, model_config):
        task_type = model_config['task_type']
        if task_type == 'classification':
            self.task_type = task_type
            self.embedding_model = model_config['embedding_model']
            self.n_neighbors = model_config['n_neighbors']
            self.training_data = None
            self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            self.instances = None
            self.labels = None
        else:
            raise ValueError(f'Model is not supported for {task_type} task.')
        
    def train(self, training_config):
        raise RuntimeError('train is not defined for SimpleKNN')
    
    def load(self, parameters_config):
        if 'training_split_data_filename' in parameters_config:
            training_data_filename = parameters_config['training_split_data_filename']
        else:
            training_data_filename = parameters_config['training_data_filename'].format(embedding_model=self.embedding_model)
        self.training_data = pd.read_csv(training_data_filename, sep='\t')
        self.training_data["embedding"] = self.training_data.embedding.apply(lambda x: ast.literal_eval(x))
        self.instances = np.array(self.training_data["embedding"].tolist())
        self.labels = np.array(self.training_data["Labels"].tolist())
        #_, self.int_labels = np.unique(self.labels, return_inverse=True)
        self.knn.fit(self.instances, self.labels)

    def save(self, parameters_config):
        raise RuntimeError('save is not defined for SimpleKNN')
    
    def single_instance_inference(self, instance):
        predicted_label = self.knn.predict(np.array(ast.literal_eval(instance['embedding'])).reshape(1, -1))[0]
        print(instance.name, 'tweet: ', instance['Texts'])
        print('predicted label: ', predicted_label)
        return predicted_label

    def inference(self, inference_config):
        if 'labelled_test_data_filename' in inference_config:
            test_data_filename = inference_config['labelled_test_data_filename']
        else:
            test_data_filename = inference_config['test_data_filename'].format(embedding_model=self.embedding_model)
        if 'labelled_output_filename' in inference_config:
            output_filename = inference_config['labelled_output_filename']
        else:
            output_filename = inference_config['output_filename'].format(task_type=self.task_type, embedding_model=self.embedding_model, n_neighbors=self.n_neighbors)
        test_data = pd.read_csv(test_data_filename, sep='\t')
        if 'Labels' in test_data:
            test_data['gold'] = test_data['Labels']
        test_data["Labels"] = test_data.apply(lambda x: self.single_instance_inference(x), axis=1)
        if 'gold' in test_data:
            test_data[['ID', 'gold', 'Labels']].to_csv(output_filename, sep='\t', index=False)
        else:
            test_data[['ID', 'Labels']].to_csv(output_filename, sep='\t', index=False)