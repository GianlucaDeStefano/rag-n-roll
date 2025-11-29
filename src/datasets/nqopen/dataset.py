"""
This script is used to create a dataset to evaluate end-to-end RAG pipelines starting from the NQ and NQ-Open datasets.

The NQ dataset is a dataset of real user queries issued to the Google search engine associated with a Wikipedia page that contains the answer to the query.
The NQ-Open dataset is a subset of the Natural Questions dataset that contains queries whose answers are shorter than 5 tokens.

What we do here is mapping back the queries and answers from the NQ-Open dataset to the documents in the validation split of the NQ dataset to create a dataset that ccontains the queires and answers from NQopne but the documents from NQ and that be used to evaluate end-to-end RAG pipelines on real world, long form documents.
If generate_independent_queries is set to True, we take care to include queries that do not share the same document with any other query in the dataset in order to avoid redundancy. 
Additionally, we add a number of distractor documents to the dataset to simulate a more realistic scenarion in which the information needed to answer a query is hidden in a large corpus of documents most of which are not relevant to the query.
"""
import logging
from multiprocessing import Pool
import pathlib
import os
import re
import json 
import gzip
import hashlib
import random
from src.pipelines.evaluate.utilities import verify_presence
random.seed(1)

import argparse
import shutil
from datasets import load_dataset
import requests
from src.datasets.base_dataset import Dataset, CorpusObj, QueryObj, validate_dataset
logger = logging.getLogger(__name__)
from tqdm import tqdm
from bs4 import BeautifulSoup
import gdown

threads = 16

def _clean_text(text):
    # Remove double spaces
    text = re.sub(' +', ' ', text)

    # Remove empty lines between paragraphs 
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove citations between []
    text = re.sub(r'\[[^\]]+\]', '', text)    

    return text

def _prepare_nq_doc(example):
    """
    Function that inspect examples from the NQ datataset and reeturns a  document object (a dictionary) with the following fields:
        - id: id that uniquely identifies the document (In this case we'll use the hash of the html content),
        - text: content of the document,
        - related_queries: lsit of queries related to the dcoument (initially empty will be filled later),
        - url: url of the document,
        - is_adversarial: flag that indicates if the document is adversarial or not (initially False),
        - optimization_type: type of optimization applied to the document (initially None),
        - original_document: id of the original document (initially the same as the id of the document) 

    Args:
        example (dict): example from the NQ dataset

    Returns:
        dict: document object
    """
    
    # Extract document text from the html
    document_text = "\n ".join([p.text for p in BeautifulSoup(example['document']['html'],features="html.parser").find('div',{"id":'mw-content-text'}).findChildren("p", recursive=True) if p])
    
    # Remove double spaces
    document_text = _clean_text(document_text)
    
        # Remove citations between []
    document_text = re.sub(r'\[\d+\]', '', document_text) 
    
    # The id associated to a document is the hash of the document_text
    doc_id = str(hashlib.sha256(document_text.encode('utf-8')).hexdigest())
    
    return { 
        'id': doc_id,
        'title':example['document']['title'],
        'text': document_text,
        'related_queries':[],
        'url':str(example['document']['url']),
        'is_adversarial': False,
        'optimization_type': None,
        'original_document_id': example['id'],
        'question': example['question']['text'].strip(),
        "type":'dataset'
    }
     
def _prepare_nq_query(example):
    """
    Function that inspect examples from the NQ-Open datataset and reeturns a query object (a dictionary) with the following fields:

        - id: id that uniquely identifies the query,
        - question: question text,
        - answers: list of answers,
        - relevant_docs: list of ids of relevant documents containing the answer (initially empty),
        - adversarial_answers: adversarial answer generated for the query (initially empty),
    
    Args:
        example (dict): example from the NQ-Open dataset
    """
    
    query_id = str(hashlib.sha256(example['question'].encode('utf-8')).hexdigest())
    
    return {
        'id': query_id,
        'question': example['question'],
        'answers': example['answers'],
        'relevant_docs': [],
        'adversarial_answers': [],
        'metadata': {'chunked_long_answer': _clean_text(example['nq_annotated_gold']['chunked_long_answer'])}
    }
    
def process_nqopen_sample(line): 
    
    query_data = json.loads(line)
    query = _prepare_nq_query(query_data)
    query['raw'] = query_data
    return query
    
def make_nqopen_dataset(n_queries = None, n_distractor_docs=4000, generate_independent_queries=True,**kwargs): 
    """
        This function: 
            - Downloads all the components of the NQ-open dataset
            - Unzip and formats them
            - Elaborates the data to create the final dataset
    """
    
    # Load the NQ dataset from the Huggingface hub
    nq_dataset = load_dataset("natural_questions",trust_remote_code=True)
    
    # Prepare the documents and the queries formatting them as dictionaries
    docs_dataset = nq_dataset['validation'].map(lambda x : _prepare_nq_doc(x), num_proc=threads).select_columns(['id','title','text','url','is_adversarial','optimization_type','original_document_id','related_queries','question'])
    
    # Filter out the documents that have no text
    docs_dataset = docs_dataset.filter(lambda x: len(x['text']) > 0)    
    
    # Construct a dictionary that indexes the samples by their question
    queries_docs = {}
    for doc in docs_dataset:
        assert doc['question'] not in queries_docs, f"Duplicate question {doc['question']}"
        queries_docs[doc['question']] = doc

    directory = "/".join(__file__.split('/')[:-1])
    
    if not os.path.exists(f'{directory}/nq-open-oracle.jsonl.gz'):
        response = requests.get('https://github.com/nelson-liu/lost-in-the-middle/raw/main/qa_data/nq-open-oracle.jsonl.gz', stream=True)
        with open(f'{directory}/nq-open-oracle.jsonl.gz', 'wb') as f_out:
            shutil.copyfileobj(response.raw, f_out)
    
    nqopen_samples = [f for f in gzip.open(f'{directory}/nq-open-oracle.jsonl.gz')]
    
    # Set of inserted documents titles and urls used for removing queries related to different versions of the same document
    inserted_docs_titles = set()
    inserted_docs_urls = set()
    
    with Pool(4) as p:
        nqopen_queries = list(tqdm(p.imap(process_nqopen_sample, nqopen_samples),total=len(nqopen_samples)))
    
    # Shuffle list of nqopen queries
    random.shuffle(nqopen_queries)
    
    # Dictionaryies containing the refined documents and queries that will endup in the dataset
    documents = {}
    queries = {}
    
    for nqopen_query in nqopen_queries:
        
        # Make sure the query is in the original NQ open dataset and that it is not already in the refined dataset
        assert nqopen_query['question'] in queries_docs, f"Query {nqopen_query['question']} not found in the original NQ open dataset validation split"
        assert nqopen_query['id'] not in queries, f"Query {nqopen_query['id']} already in the dataset"
        
        # Get the original NQ sample from which the NQopen sample was generated
        document = queries_docs[nqopen_query['question']]
        
        # Avoid to add dupliated aricles (if generate_independent_queries is set to True)
        if generate_independent_queries and document['id'] in documents:
            continue
        
        # Make sure that the document contains the paragraph marked as 'golden' this is done to avoid including documents whose answer was cut away by preprocessing
        if not verify_presence(nqopen_query['metadata']['chunked_long_answer'],[document['text']]):
            continue
        
        # Update the sets used to check for duplicates
        inserted_docs_titles.add(document['title'])
        inserted_docs_urls.add(document['url'])
        
        # Link the document to the query
        nqopen_query['relevant_docs'].append(document['id'])
        
        # Add the document to the dataset
        if generate_independent_queries:
            documents[document['id']] = document
        
        # Link the query to the document
        documents[document['id']]['related_queries'] += [nqopen_query['id']]
                
        # Save the query in the queries dictionary
        queries[nqopen_query['id']] = nqopen_query

        if n_queries and len(queries) >= n_queries:
            break

    # At this point the dataset is populated with the queries and all documents related to them
    # Here we include the documents that are not related to any
    distractor_docs_c = 0
    for distractor_nq_doc in tqdm(nq_dataset['train'].shuffle(seed=1),desc='Adding distractor documents ... ', total=n_distractor_docs):
        
        # Prepare the document
        distractor_doc = _prepare_nq_doc(distractor_nq_doc)
        
        if not distractor_doc['text']:
            continue
        
        # Avoid to add distractor documents related to any if the queries/corpuses already in the dataset
        if distractor_doc['id'] in documents or distractor_doc['title'] in inserted_docs_titles or distractor_doc['url'] in inserted_docs_urls:
            continue
        
        # Add the document to the dataset   
        documents[distractor_doc['id']] = distractor_doc
        
        distractor_docs_c += 1
        if distractor_docs_c >= n_distractor_docs:
            break
    
    print(f"Loaded {len(documents)} documents")
    print(f"Loaded {len(queries)} queries")
    
    # Convert all documents and queries to the CorpusObj/QueryObj format
    documents = {k:CorpusObj(
        id=v['id'],
        title=v['title'],
        text=v['text'],
        related_queries=v['related_queries'],
        is_adversarial=v['is_adversarial'],
        optimization_type=v['optimization_type'],
        type='dataset',
        metadata={'original_nq_id':v['original_document_id'], 'url':v['url']}
        ) for k,v in documents.items()}
    
    queries = {k:QueryObj(
        id=v['id'],
        text=v['question'],
        answers=v['answers'],
        related_corpuses=v['relevant_docs'],
    ) for k,v in queries.items()}
    
    dataset = Dataset('NQopen',documents.values(),queries.values())    
    
    validate_dataset(dataset)
    
    return dataset