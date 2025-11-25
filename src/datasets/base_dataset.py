"""
This module contains the classes used to represent the datasets used in the benchmarking process.
The main classes are:
    - CorpusDocument: class representing a document in the dataset
    - Query: class representing a query in the dataset
    - Dataset: class putting everything together and providing methods to efficiently load and save the dataset
"""

import ast
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
import shutil
from typing import Optional
from dacite import from_dict
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
logger = logging.getLogger(__name__)


# Dataclass representing a document in the dataset
@dataclass
class CorpusObj:
    id: object # The id of the document
    text:str # The text of the document
    is_adversarial:bool # Boolean indicating if the document is adversarial
    type:str # Denotes the origin of the document (e.g. "dataset", "crawled"...) 
    title:Optional[str]=""# The title of the document
    optimization_type:Optional[str] = None # String indicating the technique used to create an optimized version of the document (e.g. "attack-name")
    related_queries:list[object] = field(default_factory=list) # A list of query ids that the document is related to. Has to be empty if the document is irrelevant or len(related_queries) = 1 if the document is relevant.
    parent_corpus_id:Optional[object] = None # if the document has been generated from another document, this field contains the id of its parent document.
    counter:int = 0 # Number identifying the instance of the document (e.g. the first, the second, the third...) spawned from the same parent document
    metadata: dict = field(default_factory=dict) # Additional metadata associated with the document
    opt_metadata: dict = field(default_factory=dict) # Additional metadata associated with the optimization process
    
    # Property indicating if the document is irrelevant fro the current benchmark (e.g.: is not associated with any query)
    @property
    def is_irrelevant(self):
        return len(self.related_queries) == 0
    
    def to_dict(self):
        return{
            "id": self.id,
            "text": self.text,
            "is_adversarial": bool(self.is_adversarial),
            "type": self.type,
            "title": self.title,
            "optimization_type": self.optimization_type,
            "related_queries": self.related_queries,
            "parent_corpus_id": self.parent_corpus_id,
            "counter": self.counter,
            "metadata": self.metadata,
            "opt_metadata": self.opt_metadata,
        }
            
    @staticmethod
    def from_dict( data):
        return CorpusObj(
            id = data['id'],
            text = data['text'],
            is_adversarial = bool(data['is_adversarial']),
            type = data['type'],
            title = data['title'],
            optimization_type = data['optimization_type'],
            related_queries = data['related_queries'],
            parent_corpus_id = data['parent_corpus_id'],
            counter = data['counter'],
            metadata = data['metadata']                
        )
    
    def __str__(self) -> str:
        string_repr = f"Document {self.id}:\n"
        string_repr += f"\t- Type: {self.type}\n"
        string_repr += f"\t- Is adversarial: {self.is_adversarial}\n"
        string_repr += f"\t- Optimization type: {self.optimization_type}\n"
        string_repr += f"\t- Related queries: {self.related_queries}\n"
        string_repr += f"\t- Parent corpus id: {self.parent_corpus_id}\n"
        string_repr += f"\t- Counter: {self.counter}\n"
        string_repr += f"\t- Metadata: {str(self.metadata)}\n"
        string_repr += f"\t- Optimization metadata: {str(self.opt_metadata)}\n"
        return string_repr
    
    
# Dataclass representing a query in the dataset
@dataclass
class QueryObj:
    id: object # An id identifying the question uniquely
    text:str # The text containing question to ask to the model
    answers:list = field(default_factory=list) # The ground truth benign answers to the question
    text_mutated:Optional[str] = "" # A mutated version of the question
    answers_mutated:list = field(default_factory=list)  # mutated versions of the ground truth answers used to make the evaluation more robust
    adversarial_answers:list = field(default_factory=list) # adversarial answers to the give question 
    adversarial_answers_mutated: list = field(default_factory=list) # Mutated versions of the adversarial answers
    related_corpuses:list[object] = field(default_factory=list) # A list of document ids that are related to the query
    metadata: dict = field(default_factory=dict) # Additional metadata associated with the query

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "answers": self.answers,
            "text_mutated": self.text_mutated,
            "answers_mutated": self.answers_mutated,
            "adversarial_answers": self.adversarial_answers,
            "adversarial_answers_mutated": self.adversarial_answers_mutated,
            "related_corpuses": self.related_corpuses,
            "metadata": self.metadata
        }
        
    @staticmethod
    def from_dict(data):
        return QueryObj(
            id = data['id'],
            text = data['text'],
            answers = data['answers'],
            text_mutated = data['text_mutated'],
            answers_mutated = data['answers_mutated'],
            adversarial_answers = data['adversarial_answers'],
            adversarial_answers_mutated = data['adversarial_answers_mutated'],
            related_corpuses = data['related_corpuses'],
            metadata = data['metadata']
        )
        
    def __str__(self) -> str:
        string_repr = f"Query {self.id}:\n"
        string_repr += f"\t- Text: {self.text}\n"
        string_repr += f"\t- Answers: {self.answers}\n"
        string_repr += f"\t- Mutated text: {self.text_mutated}\n"
        string_repr += f"\t- Mutated answers: {self.answers_mutated}\n"
        string_repr += f"\t- Adversarial answers: {self.adversarial_answers}\n"
        string_repr += f"\t- Mutated adversarial answers: {self.adversarial_answers_mutated}\n"
        string_repr += f"\t- Related corpuses: {self.related_corpuses}\n"
        string_repr += f"\t- Metadata: {str(self.metadata)}\n"
        return string_repr
    
class Dataset(): 
    """
    Class representing a dataset
    """
    
    def __init__(self,name, documents = [], queries = []):
        """
        Args:
            name (str): name identifying the dataset
            documents (list): list containing CorpusObjects of the dataset
            queries (list): list containing QueryObjs queries of the dataset
            document_strategy (str): strategy used to select documents from the dataset (all, benign-only, no-optimization, {attack-name})
        """
        
        # Name identifying the dataset
        self.name = name
        
        # Dictionary containing the queries of the dataset (indexed by the query id)
        assert all([isinstance(query,QueryObj) for query in queries]), "All queries must be of type QueryObj"
        self._queries = {}
        self._documents = {}
        self.optimizations = set()
        
        self.add_queries(queries)
        self.add_documents(documents)
        
        for query in self._queries.values():
            for doc_id in query.related_corpuses:
                assert doc_id in self._documents, f"Document with id {doc_id} not found in the dataset"
                if query.id not in self._documents[doc_id].related_queries:
                    self._documents[doc_id].related_queries.append(query.id)
                
                if self._documents[doc_id].optimization_type is not None and self._documents[doc_id].optimization_type not in self.optimizations:
                    self.optimizations.add(self._documents[doc_id].optimization_type)
    
    def add_documents(self, documents):
        """
        Add documents to the dataset
        """
        assert all([isinstance(doc,CorpusObj) for doc in documents]), "All documents must be of type CorpusObj"
        for i,doc in enumerate(documents):
            
            if doc.id in self._documents: 
                assert doc.optimization_type != self._documents[doc.id].optimization_type, f"Found duplicated doc_id:{doc.id} with optimization: {doc.optimization_type}"
                doc.id = f'{doc.optimization_type}_{doc.id}'
                
            assert doc.id not in self._documents, f"Document with id {doc.id} already exists in the dataset"
            self._documents[doc.id] = doc
            
            for qid in doc.related_queries:
                assert qid in self._queries, f"Query with id {qid} not found in the dataset"
                if doc.id not in self._queries[qid].related_corpuses:
                    self._queries[qid].related_corpuses.append(doc.id)

    def add_queries(self, queries):
        """
        Add queries to the dataset
        """
        assert all([isinstance(query,QueryObj) for query in queries]), "All queries must be of type QueryObj"
        
        for query in queries:
            assert query.id not in self._queries, f"Query with id {query.id} already exists in the dataset"
            self._queries[query.id] = query
    
    
    def get_documents(self, selection_strategy = 'all'):
        """
        Returns documents according to the specified strategy
        The strategy can be one of the following:
            - all: all the documents are returned
            - benign-only: only the documents containing benign contexts are loaded
            - adversarial-only: if a document has an adversarial version, that is chosed to replace the original benign one
            
        Args:
            selection_strategy (str): strategy to use to select documents
            allow_multiple_doc_versions (bool): if True, the strategies 'no-optimization' and '{attack-name}' can return multiple versions of the same document (one benign and one adversarial) , if false only the adversarial version is returned.
                NB: this parameter is ignored if the strategy is 'all'.
        
        Returns:
            list: list of documents
        """
        if selection_strategy == 'all':
            # Nb: this strategy can contain multiple version of the same document
            logger.warning("Loading all documents in the dataset")
            return self._documents.values()
        elif selection_strategy == 'benign-only':
            return [document for document in self._documents.values() if not document.is_adversarial]
        elif selection_strategy == 'adversarial-only': 
            adv_documents = [document for document in self._documents.values() if document.is_adversarial]
            assert len(adv_documents) > 0, f"No documents found for with opt strategy: {selection_strategy}"
            # Get the ids of the benign documents to exclude (when querying for only adversarial documents we don't want to include their bening versions) 
            ids_to_exclude = [document.original_bening_corpus for document in adv_documents]
            benign_docs  = self._get_benign_documents(ids_to_exclude)
            
            return adv_documents + benign_docs
        else:
            raise Exception(f"Selection strategy: {selection_strategy} not supported!")
        
    @property
    def queries(self):
        """
        Return all the queries of the dataset
        
        Returns:
            list: list of queries
        
        """
        return list(self._queries.values())


    def __str__(self):
        """
        Return a string representation of the dataset
        """
        
        output =  f"Dataset {self.name} with {len(self._documents)} documents and {len(self._queries)} queries"
        
        doc_types_c = defaultdict(int)
        ben, mal, irrelevant = 0, 0,0
        for k,doc in self._documents.items():
            if doc.is_adversarial:
                mal = mal+1
            else:
                if doc.is_irrelevant:
                    irrelevant = irrelevant +1
                else:
                    ben = ben +1
            doc_types_c[doc.optimization_type] += 1

        if len(doc_types_c.keys()) >= 1:
            output += f"\n\t- {ben} benign documents"
            output += f"\n\t- {mal} adversarial documents"
            output += f"\n\t- {irrelevant} irrelevant documents"
            output += f"\n\t- Documents by optimization type:"
            for k,v in doc_types_c.items():
                output += f"\n\t\t- {k}: {v}"
        
        return output
            
    
    def save(self, folder_path, overwrite = False): 
        """
        Save the dataset in a folder. 
        This mathod will save the dataset into two files: 
            - corpus.parquet: containing the documents
            - queries.parquet: containing the queries
        """

        os.makedirs(folder_path, exist_ok=True)
        
        if overwrite and os.path.exists(f'{folder_path}/corpus.csv'):
            os.remove(f'{folder_path}/corpus.csv')
        
        if overwrite and os.path.exists(f'{folder_path}/queries.csv'):
            os.remove(f'{folder_path}/queries.csv')

        assert not os.path.exists(f"{folder_path}/corpus.csv"), f"Corpus file already exists already exists"
        assert not os.path.exists(f"{folder_path}/queries.csv"), f"Queries file already exists already exists"
                
        # Save the corpuses as a jsonl file
        def save_corpuses(path,docs):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                for doc in tqdm(docs):
                    f.write(json.dumps(doc.to_dict()) + '\n')
        
        
        # group corpuses by optimization type
        opt_types = defaultdict(list) # opt_type -> list of documents
        base_dataset = []
        for doc in self._documents.values():
            opt_types[doc.optimization_type].append(doc)
            
        for opt_type, docs in opt_types.items():
            if opt_type is None:
                save_corpuses(f"{folder_path}/corpus.jsonl", docs)
            else:
                save_corpuses(f"{folder_path}/optimizations/corpus_{opt_type}.jsonl", docs)

        # Save the queries in a jsonl file
        with open(f"{folder_path}/queries.jsonl", 'w') as f:
            for query in tqdm(self._queries.values()):
                f.write(json.dumps(query.to_dict()) + '\n')

    @staticmethod
    def load(folder_path):
        """
        Load the dataset from json format
        """
        
        assert os.path.isdir(folder_path), f"Path {folder_path} is not a folder"
        assert os.path.exists(folder_path), f"Folder {folder_path} does not exist"
                
        # Load the documents
        corpuses = []
        with open(f"{folder_path}/corpus.jsonl", 'r') as f:
            for line in tqdm(f):
                corpuses.append(CorpusObj.from_dict(json.loads(line)))
        loaded_optimizations = set()
        
        # Load additional documents (external from the initial dataset)
        if os.path.exists(f"{folder_path}/additional_corpuses"):
            for additional_corpus in os.listdir(f"{folder_path}/additional_corpuses"):
                
                if not additional_corpus.endswith('.json'):
                    continue
                
                optimization_name = additional_corpus.split('.')[0]

                assert optimization_name not in loaded_optimizations, f"Optimization {optimization_name} already loaded"
                loaded_optimizations.add(optimization_name)
                
                with open(f"{folder_path}/additional_corpuses/{additional_corpus}", 'r') as f:
                    for line in f:
                        corpuses.append(CorpusObj.from_dict(json.loads(line)))
                
        # Load the queries
        queries = []
        with open(f"{folder_path}/queries.jsonl", 'r') as f:
            for line in tqdm(f):
                queries.append(QueryObj.from_dict(json.loads(line)))
                
        # Return the dataset
        return Dataset(os.path.basename(folder_path),corpuses, queries)
    
    @property
    def hash(self):
        """
        Return the hash of the dataset
        """
        
        name_hash = hashlib.sha256(self.name.encode()).hexdigest()
        documents_hash = hashlib.sha256(b''.join([str(d).encode("utf-8") for d in self._documents.values()])).hexdigest()
        query_hash = hashlib.sha256(b''.join([str(q).encode("utf-8") for q in self._queries.values()])).hexdigest()
        return hashlib.sha256((name_hash + documents_hash + query_hash).encode()).hexdigest()
    


def validate_dataset(dataset): 
    """
        This function is used to make sure that there are not inconsistencies in the dataset. 
        It checks that:
            - every query's related documents is present in the dataset
            - every relevant document's related queries is present in the dataset
            - there are no duplicate 'counter' values with the same is_andversarial and original_bening_corpus_id
    """
    
    for qid in dataset._queries:
        for doc_id in dataset._queries[qid].related_corpuses:
            assert doc_id in dataset._documents, f"Document with id {doc_id} not found in the dataset"
    
    for doc_id, doc in dataset._documents.items():
        for qid in doc.related_queries:
            assert qid in dataset._queries, f"Query with id {qid} not found in the dataset"
    
    doc_to_counter_ben = defaultdict(list)
    doc_to_couter_mal = defaultdict(list)      
    for doc_id, doc in dataset._documents.items():
        if doc.parent_corpus_id:
            if not doc.is_adversarial:
                assert doc.counter not in doc_to_counter_ben[doc.parent_corpus_id], f"Duplicate counter value for benign document {doc_id} with original_corpus_id {doc.parent_corpus_id}"
                doc_to_counter_ben[doc.parent_corpus_id].append(doc.counter)
            else:
                assert doc.counter not in doc_to_couter_mal[doc.parent_corpus_id], f"Duplicate counter value for adversarial document {doc_id} with original_corpus_id {doc.parent_corpus_id}"
                doc_to_couter_mal[doc.parent_corpus_id].append(doc.counter)
        
    return True