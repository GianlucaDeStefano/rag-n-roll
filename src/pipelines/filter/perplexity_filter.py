
from src.pipelines.filter.base_filter import BaseFilter
from src.pipelines.evaluate.utilities import compute_perplexities, compute_perplexity_scores
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class PerplexityFilter(BaseFilter):
    """
    This filter uses the perplexity of the text to determine if it should be filter out. or not
    Documents with a perplexity higher than the threshold will be filtered out.
    """
    
    
    def __init__(self, config, device='cuda'):
        super().__init__(config)
        self.threshold = config['threshold']
        
        if config.model.startswith('gpt2'):
            self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
            self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        else:
            raise NotImplementedError(f"Model {config.model} not implemented")

    
    def filter(self, doc):
        perplexity = compute_perplexity_scores(self.model, self.tokenizer, [doc.page_content], stride=512, device='cuda')[0]
        doc.metadata['perplexity'] = perplexity
        return perplexity > self.threshold
    

