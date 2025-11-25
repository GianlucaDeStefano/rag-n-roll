
from collections import defaultdict
from dataclasses import dataclass
import json
import os
import pathlib
from typing import Optional
from tqdm import tqdm
from src.datasets.poisonrag.dataset import make_poisonrag_dataset
from src.datasets.nqopen.dataset import make_nqopen_dataset
from src.datasets.base_dataset import Dataset
import pandas as pd    

# Get path of current directory
directory = pathlib.Path(__file__).parent.resolve()

# Get all subdirectories in current directory (i.e. all datasets)
datasets_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and (not d.startswith('.') and not d.startswith('_'))]

# Construct dictionary of datasets and their paths
AVAILABLE_DATASETS = ['NQopen', 'PoisonRAG-msmarco', 'PoisonRAG-nq', 'PoisonRAG-hotpotqa']

@dataclass
class DatasetConfig:
    """
    Config of the dataset to use
    """ 
    # Name of the dataset 
    name : str
    
    # Path from where to load the dataset
    path: Optional[str] = None

    # Use the 'mutated' version of the query present in the dataset
    use_mutated_questions: bool = False
    
    # This flag indicates whether the pipeline should be tested on a dataset containing benign contexts
    include_benign: bool = True
    
    # This flag indicates whether the pipeline should be tested on a dataset containing adversarial contexts
    include_adversarial: bool = True
    
    # This flag indicates whether the pipeline should be tested on a dataset containing irrelevant information
    include_irrelevant: bool = True
    
    # This flag indicates whether the pipeline should be tested on a dataset containing only adv contexts specific to the query under test (i.e. the adversarial context is specific to the query)
    include_only_query_specific_adversarial_docs: bool = False
    
    # This flags controls the maximum number of benign documents included in the dataset for the given query. 
    num_benign_docs: int = 1
    
    # This flags controls the maximum number of adversarial documents included in the dataset for the given query. 
    num_malicious_docs: int = 1
    
    # This field allows to specify what type of adversarial contexts should be used to test the pipeline. Currently supported values are:
    # - unoptimized: the non-optimized adversarial contexts are used
    # - PAT: the conctexts optimized using the PAT algorithm are used
    # - collision-bert: the contexts optimized using the collision-bert algorithm are used
    adv_context_type: str = "unoptimized"
    
    # This field allows to specify what types of documents to use during the benchmark
    # - all: use all documents
    # - <document_type>: type of documents to use
    document_type: str = 'all'
    

def make_dataset(dataset_name, **kwargs):
    if dataset_name.lower() == 'NQopen'.lower():
        return make_nqopen_dataset(**kwargs)
    elif dataset_name.lower() in ['poisonrag-msmarco', 'poisonrag-nq', 'poisonrag-hotpotqa']:
        return make_poisonrag_dataset(dataset_name, **kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not found. Available datasets are: {AVAILABLE_DATASETS}")

def validate_dataset(dataset: Dataset):
    
    # Make sure that: 
    # - every relevant document is connected to a single query
    # - there are no duplicate 'counter' values for a single query
    
    query_to_counter_ben = defaultdict(list)
    query_to_counter_mal = defaultdict(list)

    for doc_id, doc in dataset._documents.items():
        if doc.is_irrelevant:
            
            for query_id in doc.related_queries:
                assert query_id not in dataset._queries
                
            for query_id, query in dataset._queries.items(): 
                assert doc_id not in query.related_corpuses, f'Document: {doc_id} relevant for query:{query_id} but is_irrelevant == True '
                
            assert doc.is_adversarial == False
        else:
            assert len(doc.related_queries) == 1, f"Document {doc_id} is relevant to multiple queries"
        
        for query_id in doc.related_queries:
            
            if not doc.is_adversarial:
                if doc.counter in query_to_counter_ben[query_id]:
                    raise ValueError(f"Duplicate benign counter value {doc.counter} for query {query_id}")
                query_to_counter_ben[query_id].append(doc.counter)
            else: 
                k = f"{doc.optimization_type}_{doc.counter}"
                if k in query_to_counter_mal[query_id]:
                    raise ValueError(f"Duplicate malicious counter value {doc.counter} for query {query_id}")
                query_to_counter_mal[query_id].append(k)
                


