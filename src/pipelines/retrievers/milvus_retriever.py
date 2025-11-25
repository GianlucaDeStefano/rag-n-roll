from abc import abstractmethod
import gc
import os
from src.pipelines.retrievers.base_retriever import make_emb_function
from src.pipelines.retrievers.base_retriever import BaseRetrieverWrapper
import shutil
from tqdm import tqdm
from langchain_milvus import Milvus

import logging

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
from tqdm import tqdm 
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

class MilvusRetreiver(BaseRetrieverWrapper): 
    """
        This class defines the interface through which all retrievers should be implemented.
    """
    
    def __init__(self, config, id, persistance_folder = '/tmp',cache_root_folder = 'data/cache'):
        """
        This method initializes the retriever with the given configuration.
        
        Args:
            config (OmegaConf): configuration of the retriever
            id (str): id of the retriever (used to store the data in the cache)
            persistance_folder (str): path to the root folder where the retriever will store its data
        """
        
        super().__init__(config=config)
        
        self.id = str(id)
        
        # Instantiate the embedding function
        self.emb_function = make_emb_function(config.name)
        
        # score function name
        self.score_function = config.score_function
        assert self.score_function in ['l2', 'ip', 'cosine'], f"Score function {self.score_function} is not supported, we can test: {self.score_function}"
        
        # Reference to the cache path where the retriever will store its data
        if not os.path.exists(persistance_folder):
            os.makedirs(persistance_folder)
        
        self.local_db_folder = os.path.join(persistance_folder, self.id)
        self.local_db_path = os.path.join(self.local_db_folder,'vector.db')
        logger.info(f'Persisting db at: {self.local_db_path}')

        os.makedirs(cache_root_folder, exist_ok=True)
        self.cached_db_folder = os.path.join(cache_root_folder, self.id)
        self.cached_db_path = os.path.join(self.cached_db_folder ,'vector.db')
        
        # If a db is available in the cache, we copy it to the local folder
        if not os.path.exists(self.local_db_folder) and os.path.exists(self.cached_db_folder) and os.path.exists(self.cached_db_path.replace('vector.db','is_cache_ready')):
            logger.info(f"Copying db from cache {self.cached_db_path}")
            shutil.copytree(self.cached_db_folder, self.local_db_folder)
        
        self.langchainDbWrapper = None

        # Check if the local folder contains a valid db
        if os.path.exists(self.local_db_path) and not os.path.exists(self.local_db_path.replace('vector.db','is_db_ready')):
            
            # The local folder contains a db but it is not marked as 'ready'
            # Probably the previous run was interrupted
            # We remove the local folder
            logging.info(f"Found not ready db in {self.local_db_path}, removing it")
            shutil.rmtree(self.local_db_folder)
        
        os.makedirs(self.local_db_folder, exist_ok=True)
        self.langchainDbWrapper = Milvus(
                embedding_function=self.emb_function,
                collection_name=id.replace('-','_').replace('.',''),
                connection_args={"uri": self.local_db_path},
                index_params = {'metric_type': self.score_function.upper()}
            )

        logging.info(f"Retriever initialized with id: {self.id}")

    def add_documents(self, documents,cache=True,batch_size=100000):
        """
        This method adds the given documents to the retriever.
        
        Args:
            documents (List[Document]): list of documents to add to the retriever
            cache (bool): flag indicating whether to cache the data
        """
        
        if self.count > 0:
            return
        
        _caught_exception = None
        try:
            assert self.count == 0, f"Retriever already contains {self.count} documents, we only support adding documents to an empty retriever"

            # Convert all None values to empty strings
            for i in range(len(documents)):
                for k, v in documents[i].metadata.items():
                    documents[i].metadata[str(k)] = v if v is not None else ''

            logging.info(f"Adding {len(documents)} documents to the retriever")
            for i in tqdm(range(0, len(documents),batch_size),desc='Indexing documents...'):
                
                # Get the documents and their ids
                b_documents = documents[i:i+batch_size]
                b_documents_ids = [doc.metadata['chunk_id'] for doc in b_documents]
                
                # Check that the number of documents is the same as the number of ids
                assert len(b_documents) == len(b_documents_ids), f"Number of documents ({len(b_documents)}) is different from the number of ids ({len(b_documents_ids)})"
                assert len(set(b_documents_ids)) == len(b_documents_ids), f"Number of unique ids ({len(set(b_documents_ids))}) is different from the number of ids ({len(b_documents_ids)})"
                
                # Add documents to the db
                self.langchainDbWrapper.add_documents(b_documents,ids=b_documents_ids)
                
            # At the end, we create a file called is_db_ready in the cache folder to indicate that the cache is ready to be used
            open(self.local_db_path.replace('vector.db','is_db_ready'), 'w').close()
        
        except Exception as e:
            logger.error(f'Exception while indexing data... removing Db...')
            shutil.rmtree(self.local_db_path)
            logger.error(f'Db removed')
            logger.error(f'Original exception: {e}')
            _caught_exception = e
            
        if _caught_exception:
            raise _caught_exception
            
        # Check that the cache does not already exist
        if cache and not os.path.exists(self.cached_db_folder):
            
            # We start copying the persistance folder to the cache folder
            os.system(f'cp -r {self.local_db_folder} {self.cached_db_folder}')
            
            # At the end, we create a file called is_ready in the cache folder to indicate that the cache is ready to be used
            open(self.cached_db_path.replace('vector.db','is_cache_ready'), 'w').close()

            logging.info(f"Cache created at {self.cached_db_path}")
        
        logging.info(f"Added {len(documents)} documents to the retriever, total documents: {self.count}")
        
    def get_relevant_documents(self,question, k, include_benign=True, include_adv=True, include_irrelevant=True,document_type='all', allowed_optimization_type = None, related_doc_ids = [], num_benign_docs =1, num_malicious_docs=1): 
        """
        This method retrieves and returns the relevant documents for the given query.

        Args:
            question (str): question to ansewer
            k (int): number of documents to retrieve
            include_benign (bool): flag indicating whether to include benign documents
            include_adv (bool): flag indicating whether to include adversarial documents
            allowed_optimization_types (List[str]): list of adversarial types to include in the search
        Returns:
            List[Documents]: list of relevant documents
        """
        
        # Compute where clauses
        where = self._where(include_benign, include_adv,include_irrelevant,document_type, allowed_optimization_type,related_doc_ids,num_benign_docs,num_malicious_docs)

        # We retrieve documents without using where because filtering is super slow on chroma
        results = self.langchainDbWrapper.similarity_search(question,k, expr=where)[:k]
        
        assert len(results) <= k, f"Number of retrieved documents ({len(results)}) is greater than the number of requested documents ({k})"

        return results
    
    @property
    def count(self):
        """
        This method returns the number of documents in the retriever.
        Returns:
            int: number of documents in the retriever
        """
        
        if not self.langchainDbWrapper or not self.langchainDbWrapper.col: 
            return 0
        
        return self.langchainDbWrapper.col.num_entities
        
    def _where(self, include_benign, include_adv, include_irrelevant,document_type, allowed_optimization_type, related_doc_ids = [],num_benign_docs = 1,num_malicious_docs = 1):
        """
        This function returns the search parameters to use when retrieving documents from the datastore.
        Returns:
            dict: dictionary containing the search parameters
        """
                
        if include_benign and include_adv and not allowed_optimization_type and not related_doc_ids and not include_irrelevant:
            return None
                    
        constraints = []
        
        assert include_benign or include_adv, "At least one of the two flags (include_benign, include_adv) must be true"
        
        
        if not include_benign:
            constraints+= ["is_adversarial == True or is_irrelevant == True"]
        #     constraints.append({
        #         '$or': [
        #             {'is_adversarial': True},
        #             {'is_irrelevant': True},
        #         ]
        #     })
        
        
        if not include_adv:
            constraints += ["is_adversarial == False"]
            # constraints.append({
            #         'is_adversarial': False,
            # })
        
        if not include_irrelevant: 
            constraints += ["is_irrelevant == False"]
            # constraints.append({
            #     'is_irrelevant': False
            # })
        
        if document_type and document_type != 'all':
            constraints += [f"type == '{document_type.strip()}'"]
            # constraints.append({
            #     'type': document_type.strip()
            # })
            
        
        if include_adv and allowed_optimization_type:
            constraints += [f"is_adversarial == False or optimization_type == '{allowed_optimization_type}'"]
            # constraints.append({
            #     '$or':[
            #     {'is_adversarial': False},
            #     {'optimization_type':allowed_optimization_type},
            #     ]
            # })
            
        
        if related_doc_ids: 
            related_doc_ids_str = '","'.join(related_doc_ids)
            constraints += [f'is_irrelevant == True or id in ["{related_doc_ids_str}"]']
            # constraints.append({
            #     '$or':[
            #     {'is_irrelevant': True},
            #     {'id':{'$in':related_doc_ids}},
            #     ]
            # })
            
        
        if num_benign_docs is not None: 
            constraints += [f"is_adversarial == True or counter < {num_benign_docs}"]
            # constraints.append({
            #     '$or':[
            #     {'is_adversarial': True},
            #     {'counter':{'$lt':num_benign_docs}},
            #     ]
            # })
        
        if num_malicious_docs is not None: 
            constraints += [f"is_adversarial == False or counter < {num_malicious_docs}"]
            # constraints.append({
            #     '$or':[
            #     {'is_adversarial': False},
            #     {'counter':{'$lt':num_malicious_docs}},
            #     ]
            # })
            
        return '(' +") and (".join(constraints) +')' if len(constraints)>0 else constraints[0]