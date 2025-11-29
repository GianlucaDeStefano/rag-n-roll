
from abc import ABC, abstractmethod
from src.datasets.base_dataset import CorpusObj


class BaseFilter(ABC):
    
    def __init__(self, config):
        self.config = config
        self.name = config['name']
    
    @abstractmethod
    def filter(self, doc : CorpusObj):
        raise NotImplementedError
        