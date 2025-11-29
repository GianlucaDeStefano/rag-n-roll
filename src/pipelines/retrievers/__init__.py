from src.pipelines.retrievers.milvus_retriever import MilvusRetreiver
from src.pipelines.retrievers.sparse_retreiver import SparseRetriever
from src.pipelines.retrievers.ensemble_reriever_wrapper import EnsambleRetrieverWrapper
import os
SUPPORTED_SPARSE_RETRIEVERS = ['BM25', 'TF-IDF']

def make_retriever(config,data_id, persistance_folder = '/tmp',cache_folder = 'data/cache'): 
    """
    Given a configuration, this function returns the retriever specified in the configuration.

    Args:
        config (OmegaConf): configuration of the retriever to load
        data_id (str): cahe key of the retriever (used to store the data in the cache)
        persistance_folder (str): path to the root folder where the retriever will store its data
    Returns:
        BaseRetriever: retriever specified in the configuration
    """

    persistance_folder = os.path.abspath(persistance_folder)
    cache_folder = os.path.abspath(cache_folder)

    if hasattr(config,'retrievers') and hasattr(config,'weights'):
        return make_ensamble_retriever(config, data_id, persistance_folder,cache_folder)

    elif is_sparse_retriever(config.name):
        return SparseRetriever(config)
    else:
        print(f"Making Milvus Retriever with config: {config}", flush=True)
        return MilvusRetreiver(config = config, id = data_id, persistance_folder = persistance_folder, cache_root_folder = cache_folder)


def make_ensamble_retriever(config,data_id, persistance_folder = '/tmp',cache_folder = 'data/cache'): 
    
    retrievers = []
    for retriever_conf in config.retrievers:
        
        retriever_conf.k = config.k
        retreiever_data_id = f"{data_id}_{retriever_conf.name.replace('/','-')}"
        
        retrievers += [make_retriever(config=retriever_conf,data_id=retreiever_data_id, persistance_folder=persistance_folder,cache_folder=cache_folder)]
        
    return EnsambleRetrieverWrapper(config = config, retrievers=retrievers, wheights = config.weights, c=config.c)
        
    
    
def is_sparse_retriever(retriever_name):
    return retriever_name in SUPPORTED_SPARSE_RETRIEVERS