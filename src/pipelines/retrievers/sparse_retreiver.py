from abc import abstractmethod
from src.pipelines.retrievers.base_retriever import BaseRetrieverWrapper
from langchain_community.retrievers import BM25Retriever, TFIDFRetriever

from pydantic import BaseModel, PrivateAttr


class SparseRetriever(BaseRetrieverWrapper): 
    """
        This class defines the interface through which all retrievers should be implemented.
    """

    
    def __init__(self, config):
        super().__init__(config=config)
        self.retriever  = None
        self.count_docs = 0
    
    def add_documents(self, documents):
        """
        This method adds the given documents to the retriever.
        
        Args:
            documents (List[Document]): list of documents to add to the retriever
        """

        self.retriever = make_sparse_retriever(self.config).from_documents(documents)
        self.count_docs = len(documents)
    
    def get_relevant_documents(self,question, k=1, include_benign=True, include_adv=True, include_irrelevant=True,document_type='all', allowed_optimization_type = None, related_doc_ids = [], num_benign_docs =1, num_malicious_docs=1): 

                
        # Set the number of relevant documents to retrieve
        self.retriever.k=k*10
        
        # Get thre relevant documents
        results = self.retriever.get_relevant_documents(question)
        
        if not include_benign:
            # filter out the benign documents
            results = [r for r in results if r.metadata['is_adversarial'] == True or r.metadata['is_irrelevant'] == True ]

        if not include_adv: 
            # filter out the adversarial documents
            results = [r for r in results if r.metadata['is_adversarial'] == False]

        if not include_irrelevant:
            # filter out the irrelevant documents
            results = [r for r in results if r.metadata['is_irrelevant'] == False]
        
        if document_type and document_type.strip()!='all':
            results = [r for r  in results if r.metadata['type'] == document_type.strip()]

        if related_doc_ids: 
            results = [r for r in results if r.metadata['is_irrelevant'] == True or r.metadata['id'] in related_doc_ids]
               
        if allowed_optimization_type: 
            results = [r for r in results if r.metadata['is_adversarial'] == False or r.metadata['optimization_type'] == allowed_optimization_type]

        if num_benign_docs:
            results = [r for r in results if r.metadata['is_adversarial'] == True or r.metadata['counter'] < num_malicious_docs]

        if num_malicious_docs is not None: 
            results = [r for r in results if r.metadata['is_adversarial'] == False or r.metadata['counter'] < num_malicious_docs]
        
        count_beign = 0
        count_adv = 0 
        count_irrelevant = 0
        for adv_doc in results:
            if not adv_doc.metadata['is_adversarial']:
                count_beign += 1
            if adv_doc.metadata['is_adversarial']:
                count_adv += 1
            if adv_doc.metadata['is_irrelevant']:
                count_irrelevant += 1 if adv_doc.metadata['is_irrelevant'] else 0
        
        print(count_beign,count_adv,count_irrelevant)
        
        assert len(results) >= k, f"Number of relevant documents is less than k. Number of relevant documents: {len(results)}, k: {k}"
        
        return results[:k]
    
    @property
    def count(self):
        """
        This method returns the number of documents in the retriever.
        """
        return self.count_docs
    
def make_sparse_retriever(config): 
    """
    Given a configuration, this function returns the retriever specified in the configuration.

    Args:
        config (OmegaConf): configuration of the retriever to load
    Returns:
        BaseRetriever: retriever specified in the configuration
    """
    
    if config['name'] == 'BM25':
        return BM25Retriever
    elif config['name'] == 'TF-IDF':
        return TFIDFRetriever
    else:
        raise ValueError(f"Retriever {config.name} not supported")