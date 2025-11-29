"""
This pipeline sets up a convenient framework to handle different types of pipelines pipelines.
"""
from abc import abstractmethod
from collections import defaultdict
import json
from pathlib import Path
import time
from dataclasses import dataclass
import logging
import os
from tqdm import tqdm
#from numba import cuda
from typing import Any, Dict, List
from src.pipelines.filter import make_chunk_filter
from src.pipelines.evaluate.utilities import verify_presence_multi
from src.pipelines.prompting import make_chain
from langchain_community.document_transformers import LongContextReorder
from src.pipelines.retrievers import is_sparse_retriever, make_retriever
from src.constants import SUPPORTED_OPTIMIZATIONS
from src.utilities.gpu.gpu import infer_device
import torch
from langchain_anthropic import ChatAnthropic
from langchain.docstore.document import Document
from src.datasets.base_dataset import Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from langchain_openai import ChatOpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

#from langchain_anthropic import ChatAnthropic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OVERRIDE_DATASETS = False # If yes each time a dataset is loaded it is re-initialized from scratch
logger = logging.getLogger(__name__)

class BaseRagPipeline(): 
    """
        This class defines a base pipeline that can be used to generate answers to questions on the base of some documents.
        
        This is the base implementation, which contains the most basic RAG pipeline one can think of. 
        The answer to a query is generated in the following way: 
            1) The query is used to retrieve a set of documents from the retriever
            2) The query and the retrieved documents are used to generate a prompt given to the model as a single input
            3) The model generates the answer to the query given the prompt
    """
    
    def __init__(self, conf, name, include_benign, include_adv, include_irrelevant, include_only_query_specific_adversarial_docs, optimization_type = None,document_type='all',num_benign_docs=1,num_malicious_docs=1,fetch_k = None, overwrite=True, data_persistance_folder = '/tmp/',data_cache_folder='data/cache'): 
        """
        Args:
            conf (hydra.Conf): configuration object containing the metadata of the pipeline
            data_path (str): path to the folder containing all the pipeline's data
            name (str, optional): Name of the pipeline overwriting the defult one. Defaults to None.
            overwrite (bool, optional): Should the pipeline initialization overwrite the existing one? Defaults to True.
        """
        
        self.type = "Naive"

        # Configuration object containing the metadata of the pipeline
        self._conf = conf
        
        # Name of the pipeline (if not given, it is automatically generated)
        self._name = name
        
        # Flags controlling which type of data to use
        # Should the pipeline be tested including benign documents?
        self._include_benign = include_benign 
        
        # Should the pipeline be tested including adversarial documents?
        self._include_adv = include_adv 
        
        # Should the pipeline be tested including irrelevant documents?
        self._include_irrelevant = include_irrelevant
        
        # Should the pipeline be tested including only adversarial documents specific to the query under test?
        # TO deleted
        self.include_only_one_adversarial = include_only_query_specific_adversarial_docs
        
        # Type of adversarial documents to use
        self._optimization_type = optimization_type

        # Type of documents to use
        self.document_type = document_type
        
        # Maximum number of benign documents to include in the dataset
        self.num_benign_docs = num_benign_docs
        
        # Maximum number of malicious documents to include in the dataset
        self.num_malicious_docs = num_malicious_docs
        
        # Instance of the dataset currently used by the pipeline
        self._dataset = None
        
        # Retriever object used to extract relevant information from the data given a query            
        self._retriever = None
        
        # Amount of documents to fetch at each iteration from the DB 
        self._fetch_k = fetch_k if fetch_k else conf.retriever.k
        
        # Reranker object used to re-order the documents after having retrieved them
        self._reranker = None

        # Folder where to save the data of the pipeline
        self._data_peristance_folder = data_persistance_folder

        # Data cache folder to get/save computed data for the pipeline
        self._data_cache_folder = data_cache_folder

        # Document processor object used to split the documents into chunks
        self._text_splitter = None
        
        # Model used to re-rank documents after having retrieved them
        self._reranking_model = None
        
        # Model object used to generate the answer given a query and a context
        self._model = None
        
        # LLM chain object used to combine the retriever and the model
        self.llm_chain = None
        
        # List of documents of the pipeline
        self._docs = []

        self._setup(overwrite)
    
    @property 
    def data_id(self):
        """Returns a string identiying uniquely the data characterizing the pipeline.
        Returns:
            str: the name of the pipeline
        """        

        return f"{self._conf.document_processor.name}-{self._conf.document_processor.chunk_size}-{self._conf.document_processor.overlap_size}_{self._conf.retriever.name}_{self._conf.retriever.score_function}_{self._dataset.hash[:10]}".replace('/','_')
    
    @abstractmethod
    def _setup(self, overwrite): 
        """
        Sets up the pipeline. 
        """        
        # Load the document processor class
        self._text_splitter = make_document_processor(self._conf.document_processor)
        
        # Load the retriever
        self._retriever = None
        
        # load the filer 
        self._filter = make_chunk_filter(self._conf.filter)
                
    def run(self,query_id, question, related_doc_ids = []): 
        """Wrapper function for the predict method of the model.

        Args:
            question (str): question to ask the model to predict
            allowed_adv_document_ids (list): list of adv_documents that 
        """
        logger.info(f"Processing query ({query_id}): {question}")      

        # If no LLM chain has been created yet, create it
        if self.llm_chain is None:
            logger.info(f"Creating LLM chain of type: {self._conf.model.chain_type}")
            assert self._conf.model.chain_type in ['stuff', 'map_reduce', 'refine', 'map_rerank'], f"Chain type '{self._conf.model.chain_type}' not supported, supported types are: ['stuff', 'map_reduce', 'refine', 'map_rerank']"
            self.llm_chain = make_chain(self._conf.model,self.model)

            # Load re-ranking model if necessary
            if self._conf.reranker:
                name = self._conf.reranker.name
                logger.info(f'Lading reranker: {name}')
                if os.path.exists(f'../models/{name.split("/")[-1]}'):
                    name = f'../models/{name.split("/")[-1]}'   
                self._reranking_model = HuggingFaceCrossEncoder(model_name=name)
                self._reranker = CrossEncoderReranker(model=self._reranking_model, top_n=200)
            
                                
        # Retrieve the relevant documents from the 
        documents = self._retriever.get_relevant_documents(question, self._fetch_k, self._include_benign, self._include_adv, self._include_irrelevant,self.document_type,self._optimization_type,related_doc_ids,self.num_benign_docs, self.num_malicious_docs)
        
        logger.info(f'Got {len(documents)} docs from the DB')
        
        # Assert that at least a document has been returned
        assert len(documents)  > 0, f"No relevant documents found for query: {question}"
         
        # Validate the selected documents
        ben_doc_ids  = []
        mal_doc_ids = []
        
        for doc in documents:
            if not self._include_benign:
                assert doc.metadata['is_adversarial'] == True or (doc.metadata['is_irrelevant']== True and self._include_irrelevant), f'Benign document found: {doc.metadata} but include_benign is set to False'
            
            if not self._include_adv:
                assert doc.metadata['is_adversarial'] == False, f"Adversarial document found: {doc.metadata} but include_adv is set to False"
            
            if not self._include_irrelevant:
                assert doc.metadata['is_irrelevant'] == False, f"Irrelevant document found: {doc.metadata} but include_irrelevant is set to False"
            
            if doc.metadata['optimization_type'] and doc.metadata['is_adversarial']: 
                assert doc.metadata['optimization_type'] == self._optimization_type, f"Adversarial type '{doc.metadata['optimization_type']}' found, but the pipeline is configured to include only adversarial documents of type '{self._optimization_type}'"
            
            if related_doc_ids and not doc.metadata['is_irrelevant']:
                assert doc.metadata['docId'] in related_doc_ids, f"Retrieved relevant document with docId: {doc.metadata['docId']}, but it is not in the allowed documents list: {related_doc_ids}"

            if self.num_benign_docs is not None and not doc.metadata['is_adversarial']:
                assert int(doc.metadata['counter']) < self.num_benign_docs, f"Retrieved benign document with docId: {doc.metadata['docId']} has conter > {self.num_benign_docs}"
            
            if self.num_malicious_docs is not None and doc.metadata['is_adversarial']:
                assert int(doc.metadata['counter']) < self.num_malicious_docs, f"Retrieved adversarial document with docId: {doc.metadata['docId']} has counter {doc.metadata['counter']} >= {self.num_malicious_docs}"
            
            if self.document_type and self.document_type != 'all':
                assert self.document_type == doc.metadata['type']
            
            if doc.metadata['is_adversarial']: 
                assert doc.metadata['is_irrelevant'] == False, f"Adversarial document found: {doc.metadata} but is_irrelevant is set to True"
                assert doc.metadata['docId'] in related_doc_ids, f"Adversarial document found: {doc.metadata} but is not in the related documents list: {related_doc_ids}"
                mal_doc_ids += [doc.metadata['docId']]
                
            if not doc.metadata['is_irrelevant'] and not doc.metadata['is_adversarial']:
                assert doc.metadata['docId'] in related_doc_ids, f"Benign document found: {doc.metadata} but is not in the related documents list: {related_doc_ids}"
                ben_doc_ids += [doc.metadata['docId']]
            
            metadata = doc.metadata
            metadata['page_content'] = doc.page_content
        
        # Check that the number of benign_docs retrieved is below num_benign_docs
        assert len(list(set(ben_doc_ids))) <= self.num_benign_docs, f"Number of distinct benign documents: {len(list(set(ben_doc_ids)))} retrieved for query: {query_id} is greater than {self.num_benign_docs} the maximum number of allowed benign documents in the dataset. Retrieved documents: {list(set(ben_doc_ids))}"

        # Check that the number of benign_docs retrieved is below num_benign_docs
        assert len(list(set(mal_doc_ids))) <= self.num_malicious_docs,  f"Number of distinct malicious documents retrieved for query: {query_id} is greater than the maximum number of allowed malicious documents in the dataset. Retrieved documents: {list(set(mal_doc_ids))}"
                
        # If a re-ranking model is available re order the retrieved chunks
        if self._reranker:
            logger.info(f"Reranking documents... ")      
            documents = self._reranker.compress_documents(
                documents, question) 
        
        documents_metadata = [doc.metadata for doc in documents]
        
        # Filter documents if a filter is available
        for doc in documents:
            doc.metadata['is_filtered_out'] = False
            doc.metadata['filter_reason'] = []
            if self._filter:
                if self._filter.filter(doc):
                    doc.metadata['is_filtered_out'] = True
                    doc.metadata['filter_reason'] = self._filter.name
 
        # Select only the first K valid documents in the list
        relevant_documents = [doc for doc in documents if not doc.metadata['is_filtered_out']][:self._conf.retriever.k]
        logger.info(f"Selected {len(relevant_documents)} relevant documents")      
        
        # We place the most relevant documents at the beginin or at the end of the prompt
        #reordering = LongContextReorder()
        #documents = reordering.transform_documents(documents)
        
        # Make sure that the number of relevant document to use to generate the answer is <= to K
        assert len(relevant_documents)  <= self._conf.retriever.k, f"Number of relevant documents found ({len(relevant_documents)}) is greater than the maximum number of documents to retrieve ({self._conf.retriever.k})"
        
        prompt_recorder = PromptsRecorder()
        
        # Generate a response using the LLM
        response = self.llm_chain.invoke(input={'input_documents':relevant_documents, 'question':question}, config={"callbacks": [prompt_recorder]})

        logger.info(f"Answer generated: {response['output_text']}")
        return response['output_text'], documents_metadata, prompt_recorder.prompts
    
    @property
    def model(self): 
        if self._model is None:
            self._model = make_model(self._conf['model'])
        return self._model
    
    def load_dataset(self, dataset: Dataset): 
        """
            Give a dataset to load its data into the pipeline.
            
            Args: 
                dataset (str): name of the dataset to load
                context_strategy (str): strategy to use to select the documents to load in the pipeline
        """    
        
        logger.info(f"Loading dataset: {dataset.name}")    
        
        # Store the dataset object and the context strategy
        self._dataset = dataset
        
        # Log the data-fingerprint
        logger.info(f"Pipeline data fingerprint: {self.data_id}")
        
        # Load the retriever
        self._retriever = make_retriever(self._conf.retriever, self.data_id, persistance_folder=self._data_peristance_folder, cache_folder=self._data_cache_folder)
        
        # IF the retriever is not empty, we assume it must be ready to use
        if self._retriever.count == 0:
            
            # Add the documents to the retriever
            logging.info('The retriever is empty, adding the documents to it.')
        
            # Convert the text into LangChain documents
            documents = []
            
            # Iterate over the documents in the dataset
            for docid, document in tqdm(dataset._documents.items(),desc='Converting documents into LangChain documents'):

                # convert each document into langchain-compatible documents
                doc = Document(page_content = document.text, metadata= {k:t for k,t in document.to_dict().items() if k not in ['text','related_queries','metadata','opt_metadata']})
                
                # we convert related_queries and opt_metadata to a string as list are not supported as metadata by DBs such as chroma
                doc.metadata['related_queries'] = json.dumps(document.related_queries)
                doc.metadata['metadata']= json.dumps(document.metadata)
                doc.metadata['opt_metadata'] = json.dumps(document.opt_metadata)
                
                # Save the property is_irrelevant explicitly
                doc.metadata['is_irrelevant'] = document.is_irrelevant
                
                # The default optimization type is unoptimized if not specified
                if not doc.metadata['optimization_type']:
                    doc.metadata['optimization_type'] = 'unoptimized'
                
                # Double check that the document only reference a single query in the Dataset
                matched_queries = 0
                for query_id in document.related_queries: 
                    if query_id in dataset._queries: 
                        matched_queries += 1
                assert matched_queries <= 1, f"Document: {docid} is matching {matched_queries} when it should match at most one"
                documents.append(doc)

            logger.info('Found {} documents to load'.format(len(documents)))
            assert len(documents) > 0, "No documents to load were found"
            
            # Split the documents into chunks
            chunks = []
            adv_documents = 0
            adversarial_splits = 0
            # Split the documents into chunks
            for doc in tqdm(documents,desc='Splitting documents into chunks'): 
                
                # Split the current document into chunks
                doc_chunks = self._text_splitter.split_documents([doc])
                
                chunk_startindex = 0
                for i,chunk in enumerate(doc_chunks): 
                    
                    chunk_opt_metadata = []
                    
                    for opt_metadata in json.loads(chunk.metadata['opt_metadata']):
                        if opt_metadata['position'] < chunk_startindex:
                            continue
                        
                        if opt_metadata['position'] > chunk_startindex + len(str(chunk.page_content))+1:
                            continue
                        
                        opt_metadata['chunk_position'] = opt_metadata['position'] - chunk_startindex
                        chunk_opt_metadata.append(opt_metadata)
                        
                    chunk.metadata['opt_metadata'] = json.dumps(chunk_opt_metadata)
                    chunk.metadata['docId'] = doc.metadata['id']
                    chunk.metadata['chunk_id'] = '{}_{}_{}'.format(doc.metadata['id'],doc.metadata['optimization_type'],i)
                    chunk.metadata['chunk_start_pos'] = chunk_startindex
                    chunk.metadata['is_golden'] = False
                    chunk_startindex += len(str(doc.page_content))
                    
                    for query_id in json.loads(chunk.metadata['related_queries']):
                        if query_id in dataset._queries: 
                            reference_query = dataset._queries[query_id]
                            if chunk.metadata['is_adversarial']: 
                                is_mal = verify_presence_multi(reference_query.adversarial_answers + reference_query.adversarial_answers_mutated,[chunk.page_content])
                                chunk.metadata['is_golden'] = is_mal
                            else: 
                                is_ben =verify_presence_multi(reference_query.answers + reference_query.answers_mutated,[chunk.page_content])
                                chunk.metadata['is_golden'] = is_ben

                    chunks.append(chunk)

                if doc.metadata['is_adversarial']: 
                    adv_documents+=1
                    adversarial_splits += len(doc_chunks)
            
            logger.info(f"Generated {len(chunks)} chunks, {adversarial_splits} of which are adversarial")
            
            # Add the documents to the retriever
            self._retriever.add_documents(chunks)
            
            assert self._retriever.count == len(chunks), f"Number of documents in the retriever ({self._retriever.count}) is different from the number of documents added ({len(chunks)})"
        else:
            logging.info('Using pre-created retriever with {} documents'.format(self._retriever.count))

        # Create a temporary retriever to add the documents to
        assert self._retriever.count > 0, f"No documents found in the computed retriever {self._retriever}"        
        
    def dispose(self):
        """
        Dispose of the pipeline and its data
        """
        del self._model
        del self._text_splitter
        del self._dataset
        del self.llm_chain
        time.sleep(5)
                
        

def make_document_processor(document_processor_config):
    """
        Return the processor to use to split the documents.
    """
    
    if document_processor_config.name == 'CharacterTextSplitter':
        return CharacterTextSplitter(chunk_size=document_processor_config.chunk_size, chunk_overlap=document_processor_config.overlap_size)
    elif document_processor_config.name == 'RecursiveCharacterTextSplitter':
        return RecursiveCharacterTextSplitter(chunk_size=document_processor_config.chunk_size, chunk_overlap=document_processor_config.overlap_size)
    else:
        raise Exception(f'Document processor type:{document_processor_config.name} not supported')

def make_model(model_config): 
    """
    Load the model object to insert into le LLM pipeline.

    Args:
        model_config (dict): config of the model to load

    Returns: instance of the model
    """
    
    model_gen_kwargs = {
        'temperature':0,
        'top_p':1,
        'do_sample': False,
    }

    if model_config.max_new_tokens > 0:
        model_gen_kwargs['max_new_tokens'] = model_config.max_new_tokens

    if model_config.temperature > 0:
        model_gen_kwargs['temperature'] = model_config.temperature
        model_gen_kwargs['do_sample'] = True
    
    if model_config.top_p < 1:
        model_gen_kwargs['top_p'] = model_config.top_p
        model_gen_kwargs['do_sample'] = True
    
    model_type = model_config.type.strip().lower()
    logger.info(f"Loading model: {model_config.name} of type: {model_type}")
    if model_type == 'openai':
        return ChatOpenAI(name=model_config.name, temperature=model_config.temperature, top_p=model_config.top_p, max_tokens=model_config.max_new_tokens if model_config.max_new_tokens > 0 else None, seed=int(os.environ['RANDOM_SEED']))
    elif model_type == 'anthropic' :
        return ChatAnthropic(model=model_config.name, temperature=model_config.temperature, top_p=model_config.top_p, max_tokens=model_config.max_new_tokens if model_config.max_new_tokens > 0 else None)
    elif model_type=='huggingface':
        
        model_name_or_path = model_config.name
        
        # If the model is available locally, load it from there
        if os.path.exists(f'../models/{model_name_or_path.split("/")[-1]}'):
            logger.info(f"Loading model from local cache: {model_name_or_path}")
            model_name_or_path = f'../models/{model_name_or_path.split("/")[-1]}'    
        
        tokenizer=AutoTokenizer.from_pretrained(model_name_or_path,token = os.environ['HF_TOKEN'],truncate=False)
        
        if tokenizer.pad_token_id is None:
            model_gen_kwargs['pad_token_id'] = tokenizer.eos_token_id

        if model_config.prompt_template and ('llama3' in model_config.prompt_template.lower() and 'instruct' in model_config.name.lower()):
            #This fix is necessary to generate correct output with llama3
            # It comes directly from the model's card https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
            model_gen_kwargs['eos_token_id'] = [tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        
        model_name_or_path = model_config.name
        
        # If the model is available locally, load it from there
        if os.path.exists(f'../models/{model_name_or_path.split("/")[-1]}'):
            model_name_or_path = f'../models/{model_name_or_path.split("/")[-1]}'
        
        device = infer_device()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            load_in_4bit=model_config.load_in_4_bit,
            load_in_8bit=model_config.load_in_8bit,
            token = os.environ['HF_TOKEN'],
            torch_dtype=torch.bfloat16 if device.lower() != 'mps' else torch.float32,
            device_map='auto'
        )
                
        p = pipeline("text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            token = model_config.access_token,
            return_full_text=False,
            **model_gen_kwargs,
        )
        
        t =  HuggingFacePipeline(model_id=model_name_or_path,pipeline=p, model_kwargs={"pretrained_model_name_or_path":model_name_or_path, "local_files_only": True}, custom_get_token_ids = tokenizer.encode)
        return t        
    else:
        raise Exception(f'Model type:{model_type} not supported')

class PromptsRecorder(BaseCallbackHandler):
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.prompts = []
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        self.prompts += prompts

def make_pipeline(config): 
    """
    Load the pipeline object to use.
    Args:
        pipeline_config (dict): config of the pipeline to load

    Raises:
        Exception: if the pipeline type is not supported

    Returns:
        LlmPipeline: instance of the pipeline
    """
    
    if config.pipeline.type == 'base':
        return BaseRagPipeline(config.pipeline, config.name, config.dataset.include_benign, config.dataset.include_adversarial, config.dataset.include_irrelevant,config.dataset.include_only_query_specific_adversarial_docs, config.dataset.adv_context_type, config.dataset.document_type,config.dataset.num_benign_docs, config.dataset.num_malicious_docs, config.fetch_k, data_persistance_folder=config.data_persistance_folder, data_cache_folder=config.data_cache_folder)
    else:
        raise Exception(f'Pipeline type:{config.pipeline.type} not supported')

    
