
from typing import List
from dataclasses import dataclass, field
import json
import logging
import os
from typing import Optional
from tqdm import tqdm

from src.pipelines.filter import BaseDocumentFilterConfig
from src.pipelines.base_rag_pipeline import make_pipeline

logger = logging.getLogger(__name__)

@dataclass
class BasePipelineComponent:
    name: str = '???'
    name_alias: Optional[str] = None

@dataclass
class ModelConfig(BasePipelineComponent): 
    """
    Config of the model to load.
    """
    name: str = "???"
    type: str = "???"
    temperature: float = 0
    top_p:float = 1
    max_new_tokens: int = 128
    access_token: Optional[str] = None
    chain_type: str = "stuff"
    load_in_4_bit: bool = False
    load_in_8bit:bool=False
    prompt_template: Optional[str] = None

@dataclass
class Phi3Model(ModelConfig): 
    name: str = "microsoft/Phi-3.5-mini-instruct"
    type: str = "huggingface"
    name_alias: str = 'phi3'

@dataclass
class LLama2_7bModel(ModelConfig): 
    name: str = "meta-llama/Llama-2-7b-chat-hf"
    type: str = "huggingface"
    load_in_4_bit: bool = False
    prompt_template: str = 'llama-chat'
    name_alias: str = 'Llama2-7B'
    
@dataclass
class LLama2_13bModel(ModelConfig): 
    name: str = "meta-llama/Llama-2-13b-chat-hf"
    type: str = "huggingface"
    load_in_4_bit: bool = False
    load_in_8bit: bool = True
    prompt_template: str = 'llama-chat'
    name_alias: str = 'Llama2-13B'

@dataclass
class LLama3_8bModel(ModelConfig): 
    name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    type: str = "huggingface"
    load_in_4_bit: bool = False
    load_in_8bit: bool = False
    prompt_template: str = 'llama3'
    name_alias: str = 'Llama3-8B'

@dataclass
class LLama3_1_8bModel(ModelConfig): 
    name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    type: str = "huggingface"
    load_in_4_bit: bool = False
    load_in_8bit: bool = False
    prompt_template: str = 'llama3'
    name_alias: str = 'meta-llama/Llama-3.1-8B-Instruct'

@dataclass
class LLama70bModel(ModelConfig): 
    name: str = "meta-llama/Llama-2-70b-chat-hf"
    type: str = "huggingface"
    load_in_4_bit: bool = False
    prompt_template: str = 'llama-chat'
    name_alias: str = 'Llama2-70B'

    

@dataclass
class GPT3Model(ModelConfig): 
    name: str = "gpt-3.5-turbo"
    type: str = "openai"
    name_alias: str = 'GPT3'


@dataclass
class GPT4Model(ModelConfig): 
    name: str = "gpt-4-turbo"
    type: str = "openai"
    name_alias: str = 'GPT4'

@dataclass
class GPT4oModel(ModelConfig): 
    name: str = "gpt-4o"
    type: str = "openai"
    name_alias: str = 'GPT4'

@dataclass
class GPTo1Model(ModelConfig): 
    name: str = "o1-preview"
    type: str = "openai"
    name_alias: str = 'GPT-o1'
    

@dataclass
class MistralModel(ModelConfig): 
    name: str = "mistralai/Mistral-7B-v0.1"
    type: str = "huggingface"
    name_alias: str = 'Mistral-7B'

@dataclass
class Mistral8x7Model(ModelConfig): 
    name: str = "mistralai/Mixtral-8x7B-v0.1"
    type: str = "huggingface"
    load_in_4_bit: bool = True
    name_alias: str = 'Mixtral-8x7B'
    
@dataclass
class GeminiModel(ModelConfig):
    name: str = "gemini-pro"
    type: str = "Gemini"


@dataclass
class ClaudeModel(ModelConfig):
    name: str = "claude-3-sonnet-20240229"
    type: str = "anthropic"
    name_alias: str = 'Sonet'

@dataclass
class BaseDocumentProcessorConfig(BasePipelineComponent):
    name: str = "???"
    chunk_size:int = 256
    overlap_size: int = 0

@dataclass
class CharacterDocSplitterConfig(BaseDocumentProcessorConfig):
    name: str = "CharacterTextSplitter"

@dataclass
class RecursiveDocSplitterConfig(BaseDocumentProcessorConfig):
    name: str = "RecursiveCharacterTextSplitter"

@dataclass
class BaseReRankerConfig(BasePipelineComponent):
    name: str = '???'

@dataclass
class MiniLmL12V2ReRankerConfig(BaseReRankerConfig):
    name:str = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
    name_alias: str = 'MiniLM'

@dataclass
class TinyBertReRankerConfig(BaseReRankerConfig):
    name:str = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
    name_alias: str = 'TinyBERT'

@dataclass
class BgeReRankerConfig(BaseReRankerConfig):
    name:str = 'BAAI/bge-reranker-base'
    name_alias: str = 'Bge'


@dataclass
class BaseRetrieverConfig(BasePipelineComponent): 
    """
    Config of the retriever to load.
    """
    k:int = 1 # number of documents to retrieve    
    score_function: Optional[str] = None
    
@dataclass
class DenseRetrieverConfig(BaseRetrieverConfig): 
    batch_size: int = 32 # batch size to use to index the selected dataset
    score_function: str = "cosine"

@dataclass
class BM25RetrieverConfig(BaseRetrieverConfig): 
    name: str = "BM25"
    name_alias : str = "BM25"

@dataclass
class AllMiniLM6V2RetrieverConfig(DenseRetrieverConfig): 
    name: str = "sentence-transformers/all-MiniLM-L6-v2"
    name_alias: str = 'MiniLM-L6'

@dataclass
class ContrieverRetrieverConfig(DenseRetrieverConfig): 
    name: str = "facebook/contriever"
    name_alias: str = 'contriever'

@dataclass
class AllMiniLM12V2RetrieverConfig(DenseRetrieverConfig): 
    name: str = "sentence-transformers/all-MiniLM-L12-v2"
    name_alias: str = 'All-MiniLM'

@dataclass
class MsmarcoMiniLML12v3RetrieverConfig(DenseRetrieverConfig): 
    name: str = "sentence-transformers/msmarco-MiniLM-L-12-v3"
    name_alias: str = 'MsMarco-MiniLM'

@dataclass
class BgeSmallRetrieverConfig(DenseRetrieverConfig): 
    name: str = "BAAI/bge-small-en-v1.5"
    name_alias: str = 'bge-small'


@dataclass
class OpenAiTextEmbeddingLargeConfig(DenseRetrieverConfig): 
    name: str = "text-embedding-3-large"
    name_alias: str = 'TEL'

@dataclass
class BaseEnsambleRetriever(BaseRetrieverConfig): 
    name: str = ""
    retrievers: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    c: int = 60

@dataclass
class HybridRetriever(BaseEnsambleRetriever):
    name:str = 'hybrid-bm25-bge'
    name_alias: str = 'Hybrid'
    retrievers: list[BaseRetrieverConfig] = field(default_factory=lambda: [
        BM25RetrieverConfig(),
        BgeSmallRetrieverConfig(),
    ])
    weights:list[float] = field(default_factory=lambda: [0.5,0.5])

@dataclass
class BasePipelineConfig: 
    """
    Config of the pipeline to load.
    """
    # Config of the model to load
    model: ModelConfig
    
    # Config of the document processor to use to split the documents
    document_processor : BaseDocumentProcessorConfig
    
    # Config of the retriever to use to index the data
    retriever : BaseRetrieverConfig
    
    # Config of the raranker to use to re-order the data
    reranker : Optional[BaseReRankerConfig] = None
    
    # Config of the document filtering system
    filter : Optional[BaseDocumentFilterConfig] = None   
    
    # Type of the pipeline to load (one of: ['base'])
    type : str = "base"
        
    # Location where data about pipelines is cached, the default value is "Data/Pipelines"
    root_data_path: str = "Data/Pipelines"
    

def run_pipeline(config, dataset, queries,output_path):
    """
    This function runs the pipeline on the given dataset and queries and saves the results in a json file.

    Args:
        config (OmegaConf): configuration file of the pipeline
        dataset (Dataset): dataset to use to test the pipeline
        queries (List[str]): subset of queries to use to test the pipeline
        output_path (str): path of the folder where to save the results
                
    Returns:
        dict: dictionary containing the results of the pipeline.
    """
    
    # Load the pipeline object
    pipeline = make_pipeline(config)
    logger.info(f"Loading the pipeline's data ...")
    
    # Add the documents to the pipeline
    pipeline.load_dataset(dataset)
    
    # List to store the responses of the  pipeline
    results = []
        
    # Iterate over the queries in the dataset
    logger.info(f"Number of queries to evaluate: {len(queries)}")
    
    # Iterate over the queries in the dataset
    for query in tqdm(list(queries)): 
        
        # Select the right question to use from the query object
        question = query.text_mutated if config.dataset.use_mutated_questions else query.text
        
        # Resolve the query
        response, documents_metadata, raw_prompts = pipeline.run(query.id, question, query.related_corpuses)
        
        # Create the result object        
        result = {
            "query": query.to_dict(),
            'mutated_query': config.dataset.use_mutated_questions,
            'k': config.pipeline.retriever.k,
            'results':{
                "relevant_documents": documents_metadata,
                "raw_prompts": raw_prompts,
                "response": response
            }
        }
           
        # Append the response to the of responses
        results.append(result)
    
    pipeline.dispose()
    
    # Save the responses into a json file 
    json.dump(results, open(os.path.join(output_path,"responses.json"), "w"), indent=4)
    return results