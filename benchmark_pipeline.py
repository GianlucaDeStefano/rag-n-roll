"""
    This script load a pipeline from a config file and tests it generating a benchmark.json file with the results.
    This file can be used to generate statistics about the pipeline using the script: evaluate_banchmark_results.py
"""
import gc
import json
import os
import time
import torch
import random
from src.pipelines.filter import PerplexityFilterConfig
from src.pipelines.evaluate import evaluate_results
from src.utilities.gpu.gpu import set_seed
from src.utilities.hydra.hydra import get_hydra_mode, get_hydra_output_dir, get_hydra_sweep_dir
from src.datasets import *
from src.pipelines import *
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from dotenv import load_dotenv
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)

set_seed(0)

@dataclass
class EvalautionConfig:
    string_matching: bool = True
    bert_score: bool = True
    ragas: bool = True
    
@dataclass
class BenchmarkConfig:

    # Configuration describing the pipeline to be tested
    pipeline: BasePipelineConfig
    
    # Configuration describing the dataset to be used to test the pipeline
    dataset: DatasetConfig

    # Configuration describing how to evalaute the results of the pipeline
    evaluation: EvalautionConfig

    # Name to give to the run, used to group experiments of the sama category in the same folder
    name: str = "default"   
    
    # Number of documents to fetch at each iteration (different from retriever.k since these are the documents retrieved and saved while k represents the total amount od documents that are inserted in the prompt)
    fetch_k: Optional[int] = None # number of documents to fetch
        
    # Folder where to persist the metadata created during the run of the pipeline
    data_persistance_folder: str = "/tmp/"
    
    # Folder where to cache the metadata created during the run of the pipeline (i.e.: the vector DB)
    # NB: If data is found in the cache folder, it is loaded from there instead of recomputing it
    data_cache_folder: str = "data/cache"
        
    # seed to use for the run
    seed: int = 0


cs = ConfigStore.instance()
cs.store(group='benchmark', name="benchmark", node=BenchmarkConfig)

cs.store(group="benchmark/pipeline", name='Base', node=BasePipelineConfig)
cs.store(group="benchmark/pipeline/model", name="Phi3", node=Phi3Model)
cs.store(group="benchmark/pipeline/model", name="LLama2_7B", node=LLama2_7bModel)
cs.store(group="benchmark/pipeline/model", name="LLama2_13B", node=LLama2_13bModel)
cs.store(group="benchmark/pipeline/model", name="LLama3_8B", node=LLama3_8bModel)
cs.store(group="benchmark/pipeline/model", name="LLama3.1_8B", node=LLama3_1_8bModel)
cs.store(group="benchmark/pipeline/model", name="GPT3", node=GPT3Model)
cs.store(group="benchmark/pipeline/model", name="GPT4", node=GPT4Model)
cs.store(group="benchmark/pipeline/model", name="GPT4o", node=GPT4oModel)
cs.store(group="benchmark/pipeline/model", name="GPTo1", node=GPTo1Model)
cs.store(group="benchmark/pipeline/model", name="Mistral", node=MistralModel)
cs.store(group="benchmark/pipeline/model", name="Mistral8x7", node=Mistral8x7Model)
cs.store(group="benchmark/pipeline/model", name="Claude3", node=ClaudeModel)
cs.store(group="benchmark/pipeline/model", name="Gemini", node=GeminiModel)

cs.store(group="benchmark/pipeline/document_processor", name="CharacterDocSplitter", node=CharacterDocSplitterConfig)
cs.store(group="benchmark/pipeline/document_processor", name="RecursiveDocSplitter", node=RecursiveDocSplitterConfig)

cs.store(group="benchmark/pipeline/reranker", name="MiniLmL12V2", node=MiniLmL12V2ReRankerConfig)
cs.store(group="benchmark/pipeline/reranker", name="TinyBert", node=TinyBertReRankerConfig)
cs.store(group="benchmark/pipeline/reranker", name="Bge-base", node=BgeReRankerConfig)

cs.store(group="benchmark/pipeline/retriever", name="BM25", node=BM25RetrieverConfig)
cs.store(group="benchmark/pipeline/retriever", name="AllMiniLM6v2", node=AllMiniLM6V2RetrieverConfig)
cs.store(group="benchmark/pipeline/retriever", name="Contriever", node=ContrieverRetrieverConfig)
cs.store(group="benchmark/pipeline/retriever", name="AllMiniLM12v2", node=AllMiniLM12V2RetrieverConfig)
cs.store(group="benchmark/pipeline/retriever", name="MsmarcoMiniLML12v3", node=MsmarcoMiniLML12v3RetrieverConfig)
cs.store(group="benchmark/pipeline/retriever", name="BgeSmall", node=BgeSmallRetrieverConfig)
cs.store(group="benchmark/pipeline/retriever", name="OpenAiTextEmbeddingLarge", node=OpenAiTextEmbeddingLargeConfig)
cs.store(group="benchmark/pipeline/retriever", name="Hybrid", node=HybridRetriever)

cs.store(group="benchmark/pipeline/filter", name="PerplexityFilter", node=PerplexityFilterConfig)

cs.store(group="benchmark/dataset", name="DefaultDatasetSettings", node=DatasetConfig)
cs.store(group="benchmark/evaluation", name="DefaultEvaluationSettings", node=EvalautionConfig)

def find_data_persistance_folder(hydra_output):
    """
    Multiple runs may re-use the same database or other metadata to avoid recomputing it.
    We need to store this data in a folder that is unique to the run if the run is a single run.
    If the run is a multirun, we save it in the sweep directory so that all the runs can access it.
    """
    run_mode = get_hydra_mode()
    if run_mode == 'RUN': 
        return hydra_output
    elif run_mode == 'MULTIRUN':
        return get_hydra_sweep_dir()
    else: 
        raise ValueError(f"Unsupported run mode: {run_mode}")

@hydra.main(version_base=None, config_path="./configs/pipelines", config_name="benchmark")
def main(config): 
    
    # set the random seed
    set_seed(config.benchmark.seed)
    
    # Load environmental variables from .env
    load_dotenv(override=True)
    
    # Get directory where to save data
    hydra_output = get_hydra_output_dir()
    logging.info(f'The data will be saved in {hydra_output}')
    
    # Add benchmark-name to data_persistance_folder
    config.benchmark.data_persistance_folder = os.path.abspath(f'{config.benchmark.data_persistance_folder}/{config.benchmark.name}')
    os.makedirs(config.benchmark.data_persistance_folder, exist_ok=True)
    
    # Backup benchmark configuration
    OmegaConf.save(config, f"{hydra_output}/config.yaml")
    
    try:
        # Compute the right dataset's path
        dataset_path = config.benchmark.dataset.path if config.benchmark.dataset.path else f'data/benchmarks/{config.benchmark.dataset.name}'
        
        # Load the dataset
        assert os.path.exists(dataset_path), f"Impossible to find dataset file at: {dataset_path}"
        dataset = Dataset.load(dataset_path)
        logger.info(f"Loaded dataset: {dataset}")
        
        # Validate dataset to make sure that it is well-formed
        validate_dataset(dataset)

        # Assert that the type of document requested is present in the dataset
        assert config.benchmark.dataset.adv_context_type in list(dataset.optimizations) + ['unoptimized'], f"Document type {config.benchmark.dataset.adv_context_type} not found in dataset"
        
        # Get the queries to run    
        queries = dataset._queries.values()
        assert len(queries) > 0, "No queries found!"
        
        # Generate results using the pipeline
        results = run_pipeline(config.benchmark, dataset,queries,hydra_output)
        
        # Analyze the results of the pipeline
        csv_stats = evaluate_results(results, config.benchmark.evaluation.string_matching, config.benchmark.evaluation.bert_score, config.benchmark.evaluation.ragas)

        # Save the results of the pipeline
        csv_stats.to_csv(f"{hydra_output}/statistics.csv")
        
        del queries, csv_stats, results, dataset
    except Exception as e:
        logging.critical(e, exc_info=True)
    
if __name__ == "__main__":
    
    main()

