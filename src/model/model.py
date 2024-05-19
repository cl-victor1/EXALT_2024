from abc import ABCMeta, abstractmethod

class Model(metaclass=ABCMeta):
    @abstractmethod
    def train(self, training_config):
        pass

    @abstractmethod
    def save(self, parameters_config):
        pass

    @abstractmethod
    def load(self, parameters_config):
        pass
    
    @abstractmethod
    def inference(self, inference_config):
        pass