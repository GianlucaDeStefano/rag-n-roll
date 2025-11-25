from dataclasses import dataclass
from src.pipelines.filter.perplexity_filter import PerplexityFilter

@dataclass
class BaseDocumentFilterConfig():
    name:str =''
    
@dataclass 
class PerplexityFilterConfig(BaseDocumentFilterConfig):
    name: str = 'perplexity'
    model: str = 'gpt2'
    threshold: float = 100

def make_chunk_filter(config):
    
    if not config:
        return None
    
    if config.name == 'perplexity':
        return PerplexityFilter(config)
    
    raise NotImplementedError(f"Filter type {config['name']} not implemented")